import os
from copy import deepcopy

import numpy as np
import torch

from cuasmrl.utils.gpu_utils import get_gpu_cc, get_mutatable_ops
from cuasmrl.utils.logger import get_logger

MEMORY_OPS, BAN_OPS = get_mutatable_ops(get_gpu_cc())

logger = get_logger(__name__)


class Sample:

    def __init__(self, kernel_section: list[str], engine):
        self.kernel_section = deepcopy(kernel_section)
        self.engine = engine

        self.candidates = []  # list of index mutable
        self.dims = None
        self._perf = None
        self.actions = []

    def __eq__(self, other):
        if not isinstance(other, Sample):
            return False
        if not len(self.kernel_section) == len(other.kernel_section):
            return False

        # an optimization for approximate equality
        for i in range(len(self.kernel_section)):
            # for i in range(1000):
            if i > len(self.kernel_section):
                break
            if not self.kernel_section[i] == other.kernel_section[i]:
                return False
        return True

    def __hash__(self):
        # approximate hash
        # concatenated_string = ''.join(self.kernel_section[:1000])
        concatenated_string = ''.join(self.kernel_section)
        return hash(concatenated_string)

    def __len__(self):
        assert self.dims is not None, f'no dims'
        return self.dims

    @property
    def perf(self):
        return self._perf

    @perf.setter
    def perf(self, value):
        self._perf = value

    def apply(self, index, action):
        lineno = self.candidates[index]
        if action == -1:
            if lineno > 0:
                self.kernel_section[lineno - 1], self.kernel_section[
                    lineno] = self.kernel_section[lineno], self.kernel_section[
                        lineno - 1]
            # self.candidates[index] -= 1
        elif action == 1:
            if lineno < len(self.kernel_section) - 1:
                self.kernel_section[lineno], self.kernel_section[
                    lineno +
                    1] = self.kernel_section[lineno +
                                             1], self.kernel_section[lineno]
            # self.candidates[index] += 1
        elif action == 0:
            pass
        else:
            assert False, f'invalid action: {action}'

    def get_mutable(self) -> list[int]:
        debug = False
        if os.getenv("SIP_DEBUG", "0") == "1":
            debug = True

        # determine which lines are possible to mutate
        # e.g. LDG, STG, and they should not cross the boundary of a label or
        # LDGDEPBAR or BAR.SYNC or rw dependencies
        # lines = []
        cnt = 0
        for i, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0] == '[':
                out = self.engine.decode(line)
                ctrl_code, _, predicate, opcode, dst, src = out
                if ctrl_code is None:
                    # a label
                    continue

                cnt += 1
                # opcode is like: LDG.E.128.SYS
                # i.e. {inst}.{modifier*}
                ban = False
                for op in BAN_OPS:
                    if op in opcode:
                        ban = True
                        break
                if ban:
                    if debug:
                        logger.warning(f'ban {ctrl_code} {opcode}')
                    continue

                for op in MEMORY_OPS:
                    if op in opcode:
                        if debug:
                            logger.info(f'mutable {ctrl_code} {opcode}')
                        self.candidates.append(i)
                        # lines.append(line)
                        break

        # dimension of the optimization problem
        self.dims = len(self.candidates)
        return self.dims, cnt

    def embedding(self, space):
        self.candidates.clear()
        embeds = np.zeros(space.shape, dtype=np.float32)
        cnt = 0
        for i, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0] == '[':
                out = self.engine.decode(line)
                ctrl_code, _, predicate, opcode, dst, src = out
                if ctrl_code is None:
                    # a label
                    continue

                op_embed = self.embed_opcode(opcode)
                embed = self.embed_ctrl_code(ctrl_code) + \
                        self.embed_predicate(predicate) + \
                        op_embed + \
                        self.embed_dst(dst) + \
                        self.embed_src(src)

                embeds[cnt] = np.array(embed, dtype=np.float32)
                cnt += 1
                if op_embed[0] == 1:
                    self.candidates.append(i)

        # unsqueeze the first dim;
        # so that it pre-appends `channel` dimension
        # resulting shape: [C, H, W]
        embeds = np.expand_dims(embeds, axis=0)
        return embeds

    def embed_ctrl_code(self, ctrl_code):
        _, read, write, yield_flag, stall_count = self.engine.decode_ctrl_code(
            ctrl_code)
        yield_flag = 1 if yield_flag == 'Y' else 0
        stall_count = int(stall_count[1:-1])
        return [yield_flag, stall_count]

    def embed_predicate(self, predicate):
        if predicate is None:
            return [0]
        return [1]

    def embed_opcode(self, opcode):
        # opcode is like: LDG.E.128.SYS
        # i.e. {inst}.{modifier*}
        memory_op = 0
        ban = False
        for op in BAN_OPS:
            if op in opcode:
                ban = True
                break

        if not ban:
            for op in MEMORY_OPS:
                if op in opcode:
                    memory_op = 1
                    break
        return [memory_op]

    def embed_dst(self, dst):
        # TODO pre scan to build a database of memory loc; then integerize?
        if dst is None:
            return [0]
        return [1]

    def embed_src(self, src):
        return [len(src)]
