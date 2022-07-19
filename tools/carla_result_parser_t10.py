import argparse
import re
import json
import csv
import os
import glob
from os import walk
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines


parser = argparse.ArgumentParser()
parser.add_argument('--results', type=str, default='./carla_results', help='Folder with json files to be parsed')


def get_infraction_coords(infraction_description):
    combined = re.findall('\(x=.*\)', infraction_description)
    if len(combined)>0:
        coords_str = combined[0][1:-1].split(", ")
        coords = [float(coord[2:]) for coord in coords_str]
    else:
        coords=["-","-","-"]
    
    return coords


def main():
    filenames = glob.glob(os.path.join(args.results, '**/*.json'), recursive=True)

    town_name = "Town10HD"

    # lists to aggregate multiple json files
    route_evaluation = []
    total_score_labels = []
    total_score_values = []
    scenario_names = []
    sum_collisions = 0
    sum_collisions_routes = 0
    finished_routes = 0
    total_routes = 0

    # aggregate files
    for f in filenames:
        with open(f) as json_file:
            evaluation_data = json.load(json_file)
            if len(evaluation_data['_checkpoint']['progress']) < 1:
                continue
                
            total_routes += evaluation_data['_checkpoint']['progress'][1]

            if evaluation_data['_checkpoint']['progress'][0] != evaluation_data['_checkpoint']['progress'][1]:
                finished_routes += evaluation_data['_checkpoint']['progress'][0]
                continue
            
            finished_routes += evaluation_data['_checkpoint']['progress'][0]

            eval_data = evaluation_data['_checkpoint']['records']
            total_scores = evaluation_data["values"]
            route_evaluation += eval_data
            
            total_score_labels = evaluation_data["labels"]
            total_score_values += [[float(score)*len(eval_data) for score in total_scores]]

            scenario_names += [str(2)]*len(eval_data)

            for record in evaluation_data['_checkpoint']['records']:
                if len(record['infractions']['collisions_vehicle']):
                    sum_collisions += len(record['infractions']['collisions_vehicle'])
                    sum_collisions_routes +=1

    if len(total_score_values) == 0:
        print("Evaluation incomplete. Exiting.")
        exit()
    
    print(f'Finished {finished_routes}/{total_routes} routes.')
    total_score_values = np.array(total_score_values)
    total_score_values = total_score_values.sum(axis=0)/len(route_evaluation)
    print('=========================')
    print(f'Collision Rate:     {sum_collisions_routes/total_routes*100:.2f}')
    print(f'Driving Score:      {total_score_values[0]:.2f}')
    print(f'Route Completion:   {total_score_values[1]:.2f}')
    print(f'Infraction Score:   {total_score_values[2]:.2f}')

    # dict to extract unique identity of route in case of repetitions
    route_to_id = {}
    for route in route_evaluation:
        route_to_id[route["route_id"]] = ''.join(i for i in route["route_id"] if i.isdigit())

    # build table of relevant information
    total_score_info = [{"label":label, "value":value} for label,value in zip(total_score_labels,total_score_values)]
    route_scenarios = [{"route":route["route_id"],
                        "town": town_name,
                        "duration": route["meta"]["duration_game"],
                        "length": route["meta"]["route_length"],
                        "score": route["scores"]["score_composed"],
                        "completion":route["scores"]["score_route"],
                        "status": route["status"],
                        "infractions": [(key,
                                        len(item),
                                        [get_infraction_coords(description) for description in item]) 
                                        for key,item in route["infractions"].items()]} 
                    for ix, route in enumerate(route_evaluation)]

    # compute aggregated statistics and table for each filter
    filters = ["route","status"]
    evaluation_filtered = {}

    for filter in filters:
        subcategories = np.unique(np.array([scenario[filter] for scenario in route_scenarios]))
        route_scenarios_per_subcategory = {}
        evaluation_per_subcategory = {}
        for subcategory in subcategories:
            route_scenarios_per_subcategory[subcategory]=[]
            evaluation_per_subcategory[subcategory] = {}
        for scenario in route_scenarios:
            route_scenarios_per_subcategory[scenario[filter]].append(scenario)
        for subcategory in subcategories:
            scores = np.array([scenario["score"] for scenario in route_scenarios_per_subcategory[subcategory]])
            completions = np.array([scenario["completion"] for scenario in route_scenarios_per_subcategory[subcategory]])
            durations = np.array([scenario["duration"] for scenario in route_scenarios_per_subcategory[subcategory]])
            lengths = np.array([scenario["length"] for scenario in route_scenarios_per_subcategory[subcategory]])

            infractions = np.array([[infraction[1] for infraction in scenario["infractions"]] 
                        for scenario in route_scenarios_per_subcategory[subcategory]])

            scores_combined = (scores.mean(),scores.std())
            completions_combined = (completions.mean(),completions.std())
            durations_combined = (durations.mean(), durations.std())
            lengths_combined = (lengths.mean(), lengths.std())
            infractions_combined = [(mean,std) for mean,std in zip(infractions.mean(axis=0),infractions.std(axis=0))]

            evaluation_per_subcategory[subcategory] = {"score":scores_combined,
                                                    "completion": completions_combined,
                                                    "duration":durations_combined,
                                                    "length":lengths_combined,
                                                    "infractions": infractions_combined} 
        evaluation_filtered[filter]=evaluation_per_subcategory

    # write output csv file
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    f = open(os.path.join(args.save_dir, 'aggregate_results.csv'),'w')
    csv_writer_object = csv.writer(f)   # Make csv writer object

    # writerow writes one row of data given as list object
    for info in total_score_info:
        csv_writer_object.writerow([item for _,item in info.items()])

    csv_writer_object.writerow([""])
    csv_writer_object.writerow(['Number of routes', len(route_evaluation)])
    csv_writer_object.writerow(['Total collisions', sum_collisions])
    csv_writer_object.writerow(['Total routes with collisions', sum_collisions_routes])
    csv_writer_object.writerow(['Collision rate', sum_collisions_routes/len(route_evaluation)*100])
    csv_writer_object.writerow([""])

    csv_writer_object.writerow(["town","scenario","infraction type","x","y","z"])
    # writerow writes one row of data given as list object
    for scenario in route_scenarios:
        for infraction in scenario["infractions"]:
            for coord in infraction[2]:
                if type(coord[0]) != str:
                    csv_writer_object.writerow([scenario["town"],
                                                infraction[0]]+coord)
    csv_writer_object.writerow([""])
    f.close()


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    args.save_dir = args.results

    main()
