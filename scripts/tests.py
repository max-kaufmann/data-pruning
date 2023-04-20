import unittest

import config
import main

dataset_tested = "tiny_imagenet"
architecture_tested = "cait_s36"


class TestAttacks(unittest.TestCase):

    def attack_errors(self, device):
        """runs through each attack, generating two adversarial examples, to test for any python errors."""
        attack_list = config.attack_list
        for a in attack_list:
            with self.subTest(a + "_python_error_check"):
                # This function runs through all the attacks, checking that they throw no errors
                n = main.parse_args()
                setattr(n, "attack", a)
                setattr(n, "num_batches", 1)
                setattr(n, "batch_size", 1)
                setattr(n, "device", device)
                setattr(n, "num_steps", 1)
                setattr(n, "dataset", dataset_tested)
                setattr(n, "architecture", architecture_tested)
                main.main(n)

    def test_attack_errors_cuda(self):
        """Run through attacks with cuda devie"""
        self.attack_errors("cuda")

    def test_attack_errors_cpu(self):
        """Run through attacks with cpu device"""
        self.attack_errors("cpu")


if __name__ == '__main__':
    unittest.main()
