import numpy as np
import json
import os

class TableConfig:
    @staticmethod
    def generate_random_table_height():
        """生成随机桌子高度，范围在0.67-0.75"""
        return np.random.uniform(0.67, 0.75)
    
    @staticmethod
    def save_table_height(episode_dir, table_height):
        """保存桌子高度到文件"""
        table_info = {
            'table_height': float(table_height)
        }
        with open(os.path.join(episode_dir, 'table_config.json'), 'w') as f:
            json.dump(table_info, f)
    
    @staticmethod
    def load_table_height(episode_dir):
        """从文件加载桌子高度"""
        config_path = os.path.join(episode_dir, 'table_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                table_info = json.load(f)
            return table_info['table_height']
        else:
            # 如果文件不存在，返回默认值
            return 0.72