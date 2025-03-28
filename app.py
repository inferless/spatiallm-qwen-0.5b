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
from pydantic import BaseModel, Field
import inferless

@inferless.request
class RequestObjects(BaseModel):
    url: str = Field(default="https://github.com/rbgo404/Files/raw/main/scene0000_00.ply")

@inferless.response
class ResponseObjects(BaseModel):
    layout_content: str = Field(default="Test output")

class InferlessPythonModel:
  def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained('manycore-research/SpatialLM-Qwen-0.5B')
        self.model = AutoModelForCausalLM.from_pretrained('manycore-research/SpatialLM-Qwen-0.5B').to("cuda")
        self.model.set_point_backbone_dtype(torch.float32)
        self.model.eval()

  def infer(self, request: RequestObjects) -> ResponseObjects:
        file_id = uuid.uuid4()
        file_name = f'scene_{file_id}.ply'
    
        response = requests.get(request.url)
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
        
        os.remove(file_name)
        return ResponseObjects(layout_content=pred_language_string)
        
  def finalize(self):
      self.model = None
