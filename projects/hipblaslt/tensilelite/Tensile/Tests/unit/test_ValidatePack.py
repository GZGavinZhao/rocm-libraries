################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

import unittest
import pytest
from rocisa.instruction import SWaitCnt, SBarrier

from Tensile.Components.CustomSchedule import ScheduleInfo, verify_lrs_and_grs
from test_CustomSchedule import create_base_kernel


class TestValidatePackBF16(unittest.TestCase):
    """
    Validate the Pack instructions present in BF16 kernels.
    Here, the pack commands map to v_perm.
    """
    def setUp(self):
        self.kernel = create_base_kernel()
        self.num_vmfma = 2 * self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]
        self.kernel["UsePLRPack"] = True

    def test_validate_simple_case_NT(self):
        """
        Simple passing case where the pack instructions are issued directly after the 
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[1, 4, 4, self.num_vmfma-1]],
            "LRA0": [[0]],
            "LRB0": [[0]],
            "PackA0": [[2]],
            "PackB0": [[2]],

            "GRA":  [[3,3]],
            "GRB":  [[3,3]],
            
            "LRA1": [[6]],
            "LRB1": [[6]],
            "PackA1": [[7]],
            "PackB1": [[7]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR0s"),
            SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Wait for GRs"),
            SBarrier(comment="For GRs"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR1s"),
        ]
        sched = ScheduleInfo(1, self.num_vmfma, optSchedule, syncCode, 2, 2)
        status, message = verify_lrs_and_grs(sched, {"kernel": self.kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

    # TODO: NN case
    # TODO: TT case
    # TODO: TN case: should be fine without packs



class TestValidatePackTF32:
    """
    TODO: Implement here later. In TF32 Pack map to different instructions, not all of which are the same.
    """
    pass