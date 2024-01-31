import torch
import joblib
from torchmetrics.classification import MulticlassAccuracy
from .base import SearchStrategyBase  # Adjust the import path based on your project structure

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.graph.transforms import quantize_transform_pass

class BruteForceSearch(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

    def search(self, search_space):

        model = search_space.rebuild_model(config, is_eval_mode)

        mg, _ = init_metadata_analysis_pass(model, None)
        mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": search_space.dummy_input})
        mg, _ = add_software_metadata_analysis_pass(mg, None)

        metric = MulticlassAccuracy(num_classes=search_space.num_classes)
        recorded_metrics = []

        # Iterate through each configuration in the search space
        for config in search_space.configurations:
            # Apply quantize transform pass
            mg_transformed, _ = quantize_transform_pass(mg, config)

            # Calculate software metrics
            software_metrics = self.compute_software_metrics(
                mg_transformed, config, is_eval_mode=True)

            # Calculate hardware metrics
            hardware_metrics = self.compute_hardware_metrics(
                mg_transformed, config, is_eval_mode=True)

            # Record the metrics
            recorded_metrics.append({
                "config": config,
                "software_metrics": software_metrics,
                "hardware_metrics": hardware_metrics
            })

        # Save the search results
        self._save_results(recorded_metrics)
        return recorded_metrics

    def _save_results(self, results):
        save_path = self.save_dir / "grid_search_results.pkl"
        with open(save_path, "wb") as f:
            joblib.dump(results, f)

