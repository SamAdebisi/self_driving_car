"""
Integration tests for configuration management
"""

import pytest
import yaml
from pathlib import Path
import tempfile

from self_driving_car.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Test configuration loader"""
    
    def test_config_loader_initialization(self, tmp_path):
        """Test config loader initialization"""
        config_content = """
network:
  input_size: 5
  output_size: 3
rl:
  gamma: 0.99
simulation:
  fps: 60
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        assert loader.config is not None
    
    def test_config_loader_get(self, tmp_path):
        """Test getting configuration values"""
        config_content = """
network:
  input_size: 5
  hidden_layers: [128, 64]
rl:
  gamma: 0.99
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        assert loader.get('network.input_size') == 5
        assert loader.get('rl.gamma') == 0.99
        assert loader.get('network.hidden_layers') == [128, 64]
    
    def test_config_loader_get_default(self, tmp_path):
        """Test getting configuration with default value"""
        config_content = """
network:
  input_size: 5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        assert loader.get('network.output_size', 3) == 3
        assert loader.get('missing.key', 'default') == 'default'
    
    def test_config_loader_validate(self, tmp_path):
        """Test configuration validation"""
        config_content = """
network:
  input_size: 5
  output_size: 3
rl:
  gamma: 0.99
  learning_rate: 0.0001
simulation:
  fps: 60
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        assert loader.validate() is True
    
    def test_config_loader_validate_missing_key(self, tmp_path):
        """Test validation with missing required key"""
        config_content = """
network:
  input_size: 5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ValueError):
            loader.validate()

