import os
from copy import deepcopy

import numpy as np

from cuasmrl.utils.gpu_utils import get_gpu_cc, get_mutatable_ops, get_st_window, check_adj_opcodes, check_ban_opcode, is_mem_op, has_hazard
from cuasmrl.utils.logger import get_logger

CC = get_gpu_cc()
MEMORY_OPS, BAN_OPS = get_mutatable_ops(CC)
MEMORY_OPS_INDEX = {op: i for i, op in enumerate(MEMORY_OPS)}
ST_WINDOW = get_st_window(CC)

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
        if action == 0:
            # push down
            if lineno > 0:

                # print(self.kernel_section[lineno - 1])
                # print(self.kernel_section[lineno])

                self.kernel_section[lineno - 1], self.kernel_section[
                    lineno] = self.kernel_section[lineno], self.kernel_section[
                        lineno - 1]

            # self.candidates[index] -= 1
        elif action == 1:
            if lineno < len(self.kernel_section) - 1:

                # print(self.kernel_section[lineno])
                # print(self.kernel_section[lineno+1])

                self.kernel_section[lineno], self.kernel_section[
                    lineno +
                    1] = self.kernel_section[lineno +
                                             1], self.kernel_section[lineno]
            # self.candidates[index] += 1
        else:
            assert False, f'invalid action: {action}'

    def get_mutable(self) -> list[int]:
        # pre-scan to obtain assembly file stats
        debug = False
        if os.getenv("SIP_DEBUG", "0") == "1":
            debug = True

        # determine which lines are possible to mutate
        # e.g. LDG, STG, and they should not cross the boundary of a label or
        # LDGDEPBAR or BAR.SYNC or rw dependencies
        # lines = []
        kernel_lineno_cnt = 0
        mem_loc = {}
        predicate_loc = {}
        opcode_loc = {}
        max_src_len = 0
        for i, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0] == '[':
                out = self.engine.decode(line)
                ctrl_code, _, predicate, opcode, dst, src = out
                if ctrl_code is None:
                    # a label
                    continue

                kernel_lineno_cnt += 1

                # integeralize memory location
                if dst not in mem_loc:
                    mem_loc[dst] = len(mem_loc)
                for s in src:
                    if s not in mem_loc:
                        mem_loc[s] = len(mem_loc)
                max_src_len = max(max_src_len, len(src))

                # ingeralize predicate
                if predicate not in predicate_loc:
                    predicate_loc[predicate] = len(predicate_loc)

                # ingeralize opcode
                if opcode not in opcode_loc:
                    opcode_loc[opcode] = len(opcode_loc)
                # opcode is like: LDG.E.128.SYS; i.e. {inst}.{modifier*}
                # if is_mem_op(CC, opcode):
                #     self.candidates.append(i)
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
        return self.dims, kernel_lineno_cnt, mem_loc, max_src_len, predicate_loc, opcode_loc

    def embedding(self, space, mem_loc, max_src_len, predicate_loc,
                  opcode_loc):
        self.candidates.clear()
        masks = []
        *_, H, W = space.shape
        embeds = np.zeros((H, W), dtype=np.float32)
        cnt = 0
        for lineno, line in enumerate(self.kernel_section):
            line = line.strip()
            # skip headers
            if len(line) > 0 and line[0] == '[':
                out = self.engine.decode(line)
                ctrl_code, _, predicate, opcode, dst, src = out
                if ctrl_code is None:
                    # a label
                    continue

                op_embed = self.embed_opcode(opcode, opcode_loc)
                embed = self.embed_ctrl_code(ctrl_code) + \
                        self.embed_predicate(predicate, predicate_loc) + \
                        op_embed + \
                        self.embed_dst(dst, mem_loc) + \
                        self.embed_src(src, mem_loc, max_src_len)

                embeds[cnt] = np.array(embed, dtype=np.float32)
                cnt += 1

                # only memory ops are considered mutable candidates
                # if is_mem_op(CC, opcode):
                if op_embed[0] > -1:  ## hacky fix in this op embedding scheme
                    self.candidates.append(lineno)
                    # TODO check bound of kernel_section?
                    mask = self._generate_mask(
                        ctrl_code,
                        opcode,
                        dst,
                        src,
                        self.kernel_section,
                        lineno,
                    )
                    masks.append(mask)

        # unsqueeze the first dim;
        # so that it pre-appends `channel` dimension
        # resulting shape: [1, 1, H, W]
        embeds = np.expand_dims(embeds, axis=0)
        return embeds, masks

    def embed_ctrl_code(self, ctrl_code):
        waits, r, w, yield_flag, stall_count = self.engine.decode_ctrl_code(
            ctrl_code)

        barr = []
        for i in range(6):
            if i in waits:
                barr.append(i / 6)  # normalize
            else:
                barr.append(-1)
        r = -1 if r[1] == '-' else int(r[1]) / 6
        w = -1 if w[1] == '-' else int(w[1]) / 6

        yield_flag = 1 if yield_flag == 'Y' else 0
        stall_count = int(stall_count[1:-1]) / 16  # normalize
        return barr + [r, w, yield_flag, stall_count]

    def embed_predicate(self, predicate, predicate_loc):
        # if predicate is None:
        #     return [-1]
        # return [predicate_loc[predicate] / len(predicate_loc)]
        if predicate is None:
            return [0]
        return [1]

    def embed_opcode(self, opcode, opcode_loc):
        # opcode is like: LDG.E.128.SYS
        # i.e. {inst}.{modifier*}
        # return [opcode_loc[opcode] / len(opcode_loc)]
        memory_op = -1
        ban = False
        for op in BAN_OPS:
            if op in opcode:
                ban = True
                break

        if not ban:
            for op in MEMORY_OPS:
                if op in opcode:
                    memory_op = MEMORY_OPS_INDEX[op]
                    # memory_op = 1
                    break
        return [memory_op]

    def embed_dst(self, dst, mem_loc):
        if dst is None:
            return [-1]

        # build
        total = len(mem_loc)
        return [mem_loc[dst] / total]

    def embed_src(self, src, mem_loc, max_src_len):
        total = len(mem_loc)
        embedding = [mem_loc[s] / total for s in src]
        diff = max_src_len - len(embedding)
        padding = [-1] * diff
        return embedding + padding

    def _generate_mask(
        self,
        ctrl_code,
        opcode,
        dst,
        src,
        kernel_section,
        lineno,
    ):
        prev_line = kernel_section[lineno - 1].strip()
        post_line = kernel_section[lineno + 1].strip()

        mask = [1, 1]  # valid to move up and down
        waits, r, w, _, self_stall_count = self.engine.decode_ctrl_code(
            ctrl_code)
        r = -1 if r[1] == '-' else int(r[1])
        w = -1 if w[1] == '-' else int(w[1])
        barrier = set(waits + [r] + [w])

        # if MemOp were to move up
        p_ctrl_code, _, _, p_opcode, p_dest, p_src = self.engine.decode(
            prev_line)
        if p_ctrl_code is None:
            # NOT move across labels
            mask[0] = 0
        # register deps
        elif p_dest in src:
            mask[0] = 0
        elif dst in p_src:
            mask[0] = 0
        # ban ops
        elif not check_ban_opcode(CC, p_opcode):
            mask[0] = 0
        elif not check_adj_opcodes(CC, p_opcode, p_dest, p_src, opcode, dst,
                                   src):
            mask[0] = 0
        else:
            # scoreboard
            p_wait, p_r, p_w, _, p_stall_count = self.engine.decode_ctrl_code(
                p_ctrl_code)
            p_r = -1 if p_r[1] == '-' else int(p_r[1])
            p_w = -1 if p_w[1] == '-' else int(p_w[1])
            p_barrier = p_wait + [p_r] + [p_w]
            # if p_r in waits or p_w in waits:
            #     mask[0] = 0
            if len(barrier.intersection(p_barrier)) != 0:
                mask[0] = 0

            # stall count
            ## for inst
            total = int(p_stall_count[1:-1])
            for i in range(1, 1 + ST_WINDOW):
                if mask[0] == 0:
                    break

                try:
                    tmp_ctrl, *_, tmp_opcode, tmp_dest, tmp_src = self.engine.decode(
                        kernel_section[lineno + i].strip())
                except:
                    # NOTE: decode gets error when (lineno + i) goes out of bounds,
                    # this is a hack to skip
                    tmp_ctrl = None
                if tmp_ctrl is None:
                    # if it is a label, don't care stall count
                    continue
                *_, stall_count = self.engine.decode_ctrl_code(tmp_ctrl)

                # move down check
                if has_hazard(CC, total, p_opcode, p_dest, p_src, tmp_opcode,
                              tmp_dest, tmp_src):
                    mask[0] = 0

                stall_count = int(stall_count[1:-1])
                total += stall_count

            ## for MemOp
            total = 0
            for i in range(2, 2 + ST_WINDOW):
                if mask[0] == 0:
                    break

                tmp_ctrl, *_, tmp_opcode, tmp_dest, tmp_src = self.engine.decode(
                    kernel_section[lineno - i].strip())
                if tmp_ctrl is None:
                    # if it is a label, don't care stall count
                    continue
                *_, stall_count = self.engine.decode_ctrl_code(tmp_ctrl)
                stall_count = int(stall_count[1:-1])
                total += stall_count

                # moveup check
                if has_hazard(CC, total, opcode, dst, src, tmp_opcode,
                              tmp_dest, tmp_src):
                    mask[0] = 0

        # if MemOp were to move down
        p_ctrl_code, _, _, p_opcode, p_dest, p_src = self.engine.decode(
            post_line)
        if p_ctrl_code is None:
            # NOT move across labels
            mask[1] = 0
        # register deps
        elif dst in p_src:
            mask[1] = 0
        elif p_dest in src:
            mask[1] = 0
        # ban ops
        elif not check_ban_opcode(CC, p_opcode):
            mask[1] = 0
        elif not check_adj_opcodes(CC, opcode, dst, src, p_opcode, p_dest,
                                   p_src):
            mask[1] = 0
        else:
            # scoreboard
            p_wait, p_r, p_w, *_ = self.engine.decode_ctrl_code(p_ctrl_code)
            p_barrier = p_wait + [p_r] + [p_w]
            if len(barrier.intersection(p_barrier)) != 0:
                mask[1] = 0

            # stall count
            total = 0
            ## for inst; move up several lines to check stall counts
            for i in range(1, 1 + ST_WINDOW):
                if mask[1] == 0:
                    break

                tmp_ctrl, *_, tmp_opcode, tmp_dest, tmp_src = self.engine.decode(
                    kernel_section[lineno - i].strip())
                if tmp_ctrl is None:
                    # if it is a label, don't care stall count
                    continue
                *_, stall_count = self.engine.decode_ctrl_code(tmp_ctrl)

                stall_count = int(stall_count[1:-1])
                total += stall_count
                # moveup check
                if has_hazard(CC, total, p_opcode, p_dest, p_src, tmp_opcode,
                              tmp_dest, tmp_src):
                    mask[1] = 0

            ## for memOp
            total = int(self_stall_count[1:-1])
            for i in range(2, 2 + ST_WINDOW):
                if mask[1] == 0:
                    break

                try:
                    tmp_ctrl, *_, tmp_opcode, tmp_dest, tmp_src = self.engine.decode(
                        kernel_section[lineno + i].strip())
                except:
                    # NOTE: decode gets error when (lineno + i) goes out of bounds,
                    # this is a hack to skip
                    tmp_ctrl = None
                if tmp_ctrl is None:
                    # if it is a label, don't care stall count
                    continue
                *_, stall_count = self.engine.decode_ctrl_code(tmp_ctrl)

                # move down check
                if has_hazard(CC, total, opcode, dst, src, tmp_opcode,
                              tmp_dest, tmp_src):
                    mask[1] = 0

                stall_count = int(stall_count[1:-1])
                total += stall_count

        return mask
