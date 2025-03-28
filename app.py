import argparse
import os
import glob
import requests
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd
from inference import preprocess_point_cloud, generate_layout
import rerun as rr
import rerun.blueprint as rrb

import uuid






class InferlessPythonModel:
  def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained('manycore-research/SpatialLM-Llama-1B')
        self.model = AutoModelForCausalLM.from_pretrained('manycore-research/SpatialLM-Llama-1B').to("cuda")
        self.model.set_point_backbone_dtype(torch.float32)
        self.model.eval()

  def infer(self, inputs):
        file_id = uuid.uuid4()
        file_name = f'scene_{file_id}.ply'
        
        response = requests.get(inputs["url"])
        response.raise_for_status()

        with open(file_name, 'wb') as file:
            file.write(response.content)

        
        config = {
            "point_cloud": file_name,
            "layout": f'scene_{file_id}.txt',
            "radius": 0.01,
            "max_points": 1000000,
            "headless": False,
            "connect": False,
            "serve": False,
            "addr": None,
            "save": f'scene_{file_id}.rrd',
            "stdout": False,
        }
        args = argparse.Namespace(**config)

        point_cloud = load_o3d_pcd(config["point_cloud"])
        point_cloud = cleanup_pcd(point_cloud)
        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # preprocess the point cloud to tensor features
        grid_size = Layout.get_grid_size()
        num_bins = Layout.get_num_bins()
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        current_directory = os.getcwd()

        # generate the layout
        layout = generate_layout(
            self.model,input_pcd,self.tokenizer,
            f'{current_directory}/code_template.txt',10,0.95,0.6,1,
        )
        layout.translate(min_extent)
        pred_language_string = layout.to_language_string()

        # pcd = load_o3d_pcd(config["point_cloud"])
        # points, colors = get_points_and_colors(pcd)


        # if os.path.splitext(args.output)[-1]:
        #     with open(args.output, "w") as f:
        #         f.write(pred_language_string)
        # else:
        #     output_filename = os.path.basename(point_cloud_file).replace(".ply", ".txt")
        #     os.makedirs(args.output, exist_ok=True)
        #     with open(os.path.join(args.output, output_filename), "w") as f:
        #         f.write(pred_language_string)

        # # parse layout_content
        # layout = Layout(layout_content)
        # floor_plan = layout.to_boxes()

        # # ReRun visualization
        # blueprint = rrb.Blueprint(
        #     rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
        #     collapse_panels=True,
        # )
        # rr.script_setup(args, "rerun_spatiallm", default_blueprint=blueprint)

        # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # point_indices = np.arange(points.shape[0])
        # np.random.shuffle(point_indices)
        # point_indices = point_indices[: args.max_points]
        # points = points[point_indices]
        # colors = colors[point_indices]
        # rr.log(
        #     "world/points",
        #     rr.Points3D(
        #         positions=points,
        #         colors=colors,
        #         radii=args.radius,
        #     ),
        #     static=True,
        # )

        # num_entities = len(floor_plan)
        # seconds = 0.5
        # for ti in range(num_entities + 1):
        #     sub_floor_plan = floor_plan[:ti]

        #     rr.set_time_seconds("time_sec", ti * seconds)
        #     for box in sub_floor_plan:
        #         uid = box["id"]
        #         group = box["class"]
        #         label = box["label"]

        #         rr.log(
        #             f"world/pred/{group}/{uid}",
        #             rr.Boxes3D(
        #                 centers=box["center"],
        #                 half_sizes=0.5 * box["scale"],
        #                 labels=label,
        #             ),
        #             rr.InstancePoses3D(mat3x3=box["rotation"]),
        #             static=False,
        #         )
        # rr.script_teardown(args)
        # s3_file = f'{MY_PREFIX}{config["save"]}'
        # s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
        #                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        #                  region_name=AWS_REGION)        
        # s3_client.upload_file(config["save"],MY_BUCKET,s3_file)

        return {"pred_language_string":f"{pred_language_string}"}
        
  def finalize(self):
      self.model = None


# app = InferlessPythonModel()
# app.initialize()
# app.infer({"url":"https://github.com/rbgo404/Files/raw/main/scene0000_00.ply"})

