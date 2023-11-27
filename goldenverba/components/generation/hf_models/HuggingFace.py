import yaml
from typing import List


class Options:
    options: List[str]
    selected_option: str


class HuggingFaceOptions:
    def __init__(self):
        self.options = self.get_options()
        self.selected_option = self.select_option()

    def get_options(self) -> List[str]:
        return yaml.safe_load(
            "goldenverba/components/generation/hf_models/hf_options.yaml"
        )

    def select_option(self, option: str = "microsoft/Orca-2-13b") -> str:
        if option == "microsoft/Orca-2-13b":
            return option
        for i, option in self.options:
            print(f"{i}: {option}")
        selected_value = input("Select an option: ")
        return self.options[selected_value]
