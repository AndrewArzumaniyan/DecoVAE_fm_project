"""Safety invariants of the shared-server supervisor; no actual signals are sent."""
import unittest
import tempfile
from pathlib import Path
from safe_queue import may_signal, health_reason, output_complete, dependencies_ready


class GuardTests(unittest.TestCase):
    def test_refuses_foreign_uid(self):
        p = (1001, 42, 42, 123)
        self.assertFalse(may_signal(42, p, p, 1002))

    def test_refuses_reused_pid(self):
        self.assertFalse(may_signal(42, (1002, 42, 42, 123), (1002, 42, 42, 124), 1002))

    def test_refuses_other_group_or_session(self):
        for p in [(1002, 43, 42, 123), (1002, 42, 43, 123)]:
            self.assertFalse(may_signal(42, p, p, 1002))

    def test_accepts_only_matching_created_group(self):
        p = (1002, 42, 42, 123)
        self.assertTrue(may_signal(42, p, p, 1002))

    def test_resource_rules(self):
        h = dict(foreign_gpu_pids=[], disk_free_gib=1000, available_gib=100,
                 cpu_c=50, cpu_median_c=50, gpu_c=50, load1=1, cpu_pressure10=0,
                 gpu_total_mib=24000, gpu_used_mib=2000)
        self.assertIsNone(health_reason(h, starting=True))
        self.assertIsNotNone(health_reason({**h, 'foreign_gpu_pids': [7]}))
        self.assertIsNotNone(health_reason({**h, 'cpu_c': 96}))
        self.assertIsNotNone(health_reason({**h, 'available_gib': 20}))
        self.assertIsNotNone(health_reason({**h, 'disk_free_gib': 90}))
        self.assertIsNotNone(health_reason({**h, 'gpu_error': 'unavailable'}))

    def test_failed_reproduction_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'check.json').write_text('{"passed": false}')
            job = {'outputs': ['check.json'], 'success_json': 'check.json'}
            self.assertFalse(output_complete(p, job))
            (p / 'check.json').write_text('{"passed": true}')
            self.assertTrue(output_complete(p, job))

    def test_partial_evaluation_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'scores.csv').write_text('partially written')
            self.assertFalse(output_complete(p, {'outputs': ['scores.csv'], 'success_json': 'done.json'}))

    def test_optional_failure_does_not_block_honest_report(self):
        j = {'after': ['components'], 'after_any': ['external']}
        self.assertFalse(dependencies_ready(j, {'components'}, {}, set()))
        self.assertTrue(dependencies_ready(j, {'components'}, {'external': 1}, set()))
        self.assertTrue(dependencies_ready(j, {'components'}, {}, {'external'}))
        self.assertFalse(dependencies_ready(j, set(), {'components': 1, 'external': 1}, set()))


if __name__ == '__main__':
    unittest.main()
