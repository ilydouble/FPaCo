import json

class LLMGuidance:
    """
    Helper class to simulate or interface with an LLM to get visual attributes for detection.
    Supports specialized mappings for APTOS2019, OCTA500, and Finger datasets.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        # Dataset-specific visual attributes mapping
        self.dataset_knowledge = {
            "APTOS2019": {
                "description": "Retinal Fundus images for DR classification.",
                "tasks": {
                    "DR": [
                        "microaneurysms and small red spots",
                        "retinal hemorrhages and bleeding spots",
                        "hard exudates and yellow deposits",
                        "cotton wool spots and soft exudates"
                    ],
                    "Severity": "grading based on microaneurysms and hemorrhages density"
                }
            },
            "OCTA500": {
                "description": "OCT Angiography for retinal disease classification.",
                "tasks": {
                    "General": [
                        "foveal avascular zone and centered capillary-free area",
                        "non-perfusion area and capillary dropout patches",
                        "neovascularization and abnormal vessel tangles"
                    ],
                    "DR": "enlarged foveal avascular zone and capillary non-perfusion",
                    "AMD": "neovascularization and flow signals in outer retina"
                }
            },
            "Finger": {
                "description": "Fingerprint images for pattern classification.",
                "tasks": {
                    "Patterns": [
                        "core point and spiral center",
                        "delta point and triangular ridge junction",
                        "bifurcation and ridge ending minutiae"
                    ],
                    "Center": "core point in center of fingerprint pattern",
                    "Delta": "triangular junction point where ridges meet"
                }
            }
        }

    def get_visual_attributes(self, dataset_name, task="General"):
        """
        Retrieves visual attributes for a specific dataset and task.
        """
        dataset = self.dataset_knowledge.get(dataset_name)
        if not dataset:
            return ["pathological and structural features"]
        
        attributes = dataset["tasks"].get(task, dataset["tasks"].get("General", ["features"]))
        if isinstance(attributes, str):
            return [attributes]
        return attributes

    def generate_dino_prompts(self, dataset_name, task="General"):
        """
        Step 1: LLM Guidance - Generates a list of specialized prompts for Grounding DINO.
        """
        attributes = self.get_visual_attributes(dataset_name, task)
        # We can return multiple prompts to be processed by DINO
        return attributes

# Example usage
if __name__ == "__main__":
    guide = LLMGuidance()
    print("APTOS Prompts:", guide.generate_dino_prompts("APTOS2019", "DR"))
    print("OCTA500 Prompts:", guide.generate_dino_prompts("OCTA500"))
    print("Finger Prompts:", guide.generate_dino_prompts("Finger", "Patterns"))
