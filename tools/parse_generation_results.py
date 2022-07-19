import argparse
import glob
import json

import numpy as np


class ResultsParser():
    def __init__(self, args):
        """
        """
        self.args = args

        self.results_files = self.parse_results_dir()
        print(f"Found {len(self.results_files)} results files...")

        with open('./tools/timings.json') as f:
            self.timings = json.load(f)

    def parse_results_dir(self):
        """
            Parse the results directory and gather the
            relevant JSON file paths from
            the "RouteScenario_*_to_*" subdirectories.
        """
        route_scenario_dirs = sorted(
            glob.glob(
                self.args.results_dir + "/**/RouteScenario_*", recursive=True
                ),
            key=lambda path: int(path.split("_")[-1]),
        )

        results_files = []
        for dir in route_scenario_dirs:
            results_files.extend(
                sorted(
                    glob.glob(dir + "/results.json")
                )
            )

        return results_files

    def parse_json_file(self, records_file):
        """
        """
        return json.loads(open(records_file).read())

    def generate_report(self):
        """
        """
        iterations = []
        num_routes = len(self.results_files)
        optim_method = str(self.args.optim_method)

        for results_file in self.results_files:
            results = self.parse_json_file(results_file)

            path_parts = results_file.split('/')
            found_traffic_density = False
            for part in path_parts:
                if 'agents_' in part:
                    traffic_density = part[-1]
                    found_traffic_density = True
            if not found_traffic_density:
                traffic_density = str(self.args.num_agents)

            if self.args.use_GPU_hours:
                factor = self.timings[optim_method][traffic_density][results["meta_data"]["town"]]
            else:
                factor=1

            if results["first_metrics"]["Collision Metric"] == 1 and results["first_metrics"]["iteration"]*factor < self.args.max_GPU_hours*60*60:
                iterations.append(results["first_metrics"]["iteration"]*factor)

        iterations.sort()

        print(f'Collision rate: {len(iterations)/num_routes}')
        if int(50*num_routes/100) < len(iterations):
            print(f't@50: {np.mean(iterations[:int(50*num_routes/100)])}')
        else:
            print(f'CR is lower than 50%, not computing t@50.')
        print('-------------------------------------------')


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--results_dir",
        type=str,
        default="./outputs/",

        help="The directory containing the scenario records and results files \
              for each of the optimized scenarios/routes for each traffic density.",
    )
    main_parser.add_argument(
        "--use_GPU_hours",
        default=1,
    )
    main_parser.add_argument(
        "--max_GPU_hours",
        default=0.05,
    )
    main_parser.add_argument(
        "--optim_method",
        default="Adam",
        choices=["Adam", "Both_Paths"]
    )
    main_parser.add_argument(
        "--num_agents",
        default=4
    )

    args = main_parser.parse_args()

    results_parser = ResultsParser(args)
    results_parser.generate_report()
