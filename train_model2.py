"""
Script de Treinamento do Modelo PPO
Usa os melhores hiperparâmetros encontrados pelo Optuna
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from scalping_environment import ImprovedScalpingEnv

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configurações de treinamento"""
    
    # Dados
    SYMBOL = "ETHUSDT"
    INTERVAL = "5m"
    N_CANDLES = 15000  # Mais dados para treinamento final
    
    # Treinamento
    TOTAL_TIMESTEPS = 2_000_000  # 2 milhões de steps
    PARALLEL_ENVS = 8  # Mais ambientes paralelos
    EVAL_FREQ = 50000  # Avaliação a cada 50k steps
    EVAL_EPISODES = 100
    
    # Checkpoints
    CHECKPOINT_FREQ = 100000  # Checkpoint a cada 100k steps
    
    # Validação
    TRAIN_RATIO = 0.8
    
    # Paths
    HYPERPARAMS_PATH = "optimization_results/best_hyperparams_fast.json"
    MODEL_SAVE_PATH = "trained_models"
    DATA_PATH = "training_data"

class DataManager:
    """Gerenciador de dados para treinamento"""
    
    def __init__(self, symbol: str = "ETHUSDT", interval: str = "5m"):
        self.symbol = symbol
        self.interval = interval
    
    def download_training_data(self, n_candles: int) -> str:
        """Baixa dados para treinamento"""
        logger.info(f"Baixando {n_candles} candles para treinamento")
        
        try:
            url = "https://api.binance.com/api/v3/klines"
            all_data = []
            limit = 1000
            end_time = None
            
            while len(all_data) < n_candles:
                params = {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "limit": min(limit, n_candles - len(all_data))
                }
                if end_time:
                    params["endTime"] = end_time
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data = data + all_data
                end_time = data[0][0] - 1
                
                if len(data) < limit:
                    break
            
            # Processa dados
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
            
            df = pd.DataFrame(all_data[-n_candles:], columns=cols)
            df[["open", "high", "low", "close", "volume"]] = \
                df[["open", "high", "low", "close", "volume"]].astype(float)
            
            # Salva arquivo
            os.makedirs(TrainingConfig.DATA_PATH, exist_ok=True)
            csv_path = os.path.join(TrainingConfig.DATA_PATH, f"{self.symbol}_{self.interval}_training.csv")
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Dados de treinamento salvos: {csv_path} ({len(df)} candles)")
            return csv_path
            
        except Exception as e:
            logger.error(f"Erro ao baixar dados: {e}")
            raise
    
    def split_data(self, csv_path: str, train_ratio: float = 0.8) -> tuple:
        """Divide dados em treino e validação"""
        df = pd.read_csv(csv_path)
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        train_path = csv_path.replace('.csv', '_train.csv')
        val_path = csv_path.replace('.csv', '_val.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        logger.info(f"Dados divididos - Treino: {len(train_df)}, Validação: {len(val_df)}")
        return train_path, val_path

def load_best_hyperparams() -> Dict:
    """Carrega melhores hiperparâmetros do Optuna"""
    try:
        if not os.path.exists(TrainingConfig.HYPERPARAMS_PATH):
            logger.warning("Hiperparâmetros não encontrados, usando padrões")
            return get_default_hyperparams()
        
        with open(TrainingConfig.HYPERPARAMS_PATH, 'r') as f:
            hyperparams = json.load(f)
        
        logger.info("Hiperparâmetros carregados do Optuna")
        return hyperparams
        
    except Exception as e:
        logger.error(f"Erro ao carregar hiperparâmetros: {e}")
        return get_default_hyperparams()

def get_default_hyperparams() -> Dict:
    """Hiperparâmetros padrão caso Optuna não tenha rodado"""
    return {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 256,
        "n_steps": 2048,
        "profit_threshold": 0.004,
        "stop_loss_threshold": 0.02,
        "max_position_time": 50,
        "transaction_cost": 0.0004,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "net_arch": "medium"
    }

def make_env(csv_path: str, env_kwargs: Dict, monitor_path: Optional[str] = None):
    """Factory para criar ambiente com monitoramento"""
    def _init():
        env = ImprovedScalpingEnv(csv_path=csv_path, live_stream=False, **env_kwargs)
        if monitor_path:
            env = Monitor(env, monitor_path)
        return env
    return _init

def create_model(env, hyperparams: Dict) -> PPO:
    """Cria modelo PPO com hiperparâmetros otimizados"""
    
    # Arquitetura da rede
    net_arch_dict = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])],
        "large": [dict(pi=[256, 256], vf=[256, 256])]
    }
    
    net_arch = net_arch_dict.get(hyperparams.get("net_arch", "medium"), net_arch_dict["medium"])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get("learning_rate", 0.0003),
        gamma=hyperparams.get("gamma", 0.99),
        batch_size=hyperparams.get("batch_size", 256),
        n_steps=hyperparams.get("n_steps", 2048),
        clip_range=hyperparams.get("clip_range", 0.2),
        ent_coef=hyperparams.get("ent_coef", 0.01),
        vf_coef=hyperparams.get("vf_coef", 0.5),
        policy_kwargs=dict(net_arch=net_arch),
        verbose=1,
        device='auto',
        tensorboard_log="./tensorboard_logs/"
    )
    
    return model

def setup_callbacks(val_env, model_save_path: str):
    """Configura callbacks para treinamento"""
    callbacks = []
    
    # Callback de avaliação
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=os.path.join(model_save_path, "best_model"),
        log_path=os.path.join(model_save_path, "eval_logs"),
        eval_freq=TrainingConfig.EVAL_FREQ,
        n_eval_episodes=TrainingConfig.EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Callback de checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainingConfig.CHECKPOINT_FREQ,
        save_path=os.path.join(model_save_path, "checkpoints"),
        name_prefix="ppo_scalping"
    )
    callbacks.append(checkpoint_callback)
    
    return callbacks

def evaluate_final_model(model: PPO, val_env, n_episodes: int = 50) -> Dict:
    """Avalia modelo final"""
    logger.info("Avaliando modelo final...")
    
    episode_rewards = []
    episode_lengths = []
    total_trades = 0
    winning_trades = 0
    
    for episode in range(n_episodes):
        obs = val_env.reset()
        episode_reward = 0
        episode_length = 0
        trades_this_episode = 0
        wins_this_episode = 0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = val_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Conta trades
            if reward != 0:
                trades_this_episode += 1
                if reward > 0:
                    wins_this_episode += 1
            
            if done.any():
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_trades += trades_this_episode
        winning_trades += wins_this_episode
    
    # Calcula métricas
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    win_rate = winning_trades / max(total_trades, 1)
    
    metrics = {
        'avg_episode_reward': avg_reward,
        'std_episode_reward': std_reward,
        'avg_episode_length': np.mean(episode_lengths),
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_episodes': n_episodes
    }
    
    logger.info(f"Avaliação final - Reward médio: {avg_reward:.6f}, "
               f"Win rate: {win_rate:.2%}, Total trades: {total_trades}")
    
    return metrics

def main():
    """Função principal de treinamento"""
    logger.info("=== INICIANDO TREINAMENTO DO MODELO ===")
    
    # Carrega hiperparâmetros
    hyperparams = load_best_hyperparams()
    logger.info(f"Hiperparâmetros: {hyperparams}")
    
    # Prepara dados
    data_manager = DataManager(TrainingConfig.SYMBOL, TrainingConfig.INTERVAL)
    
    # Baixa dados se necessário
    training_data_path = os.path.join(TrainingConfig.DATA_PATH, 
                                     f"{TrainingConfig.SYMBOL}_{TrainingConfig.INTERVAL}_training.csv")
    
    if not os.path.exists(training_data_path):
        training_data_path = data_manager.download_training_data(TrainingConfig.N_CANDLES)
    
    # Divide dados
    train_path, val_path = data_manager.split_data(training_data_path, TrainingConfig.TRAIN_RATIO)
    
    # Parâmetros do ambiente
    env_kwargs = {
        'profit_threshold': hyperparams.get('profit_threshold', 0.004),
        'stop_loss_threshold': hyperparams.get('stop_loss_threshold', 0.02),
        'max_position_time': hyperparams.get('max_position_time', 50),
        'transaction_cost': hyperparams.get('transaction_cost', 0.0004),
        'force_dim': 20
    }
    
    # Cria ambientes
    os.makedirs(TrainingConfig.MODEL_SAVE_PATH, exist_ok=True)
    monitor_path = os.path.join(TrainingConfig.MODEL_SAVE_PATH, "monitor_logs")
    os.makedirs(monitor_path, exist_ok=True)
    
    train_env = DummyVecEnv([
        make_env(train_path, env_kwargs, os.path.join(monitor_path, f"train_{i}"))
        for i in range(TrainingConfig.PARALLEL_ENVS)
    ])
    
    val_env = DummyVecEnv([
        make_env(val_path, env_kwargs, os.path.join(monitor_path, f"val_{i}"))
        for i in range(min(4, TrainingConfig.PARALLEL_ENVS))  # Menos ambientes para validação
    ])
    
    # Cria modelo
    model = create_model(train_env, hyperparams)
    
    # Configura callbacks
    callbacks = setup_callbacks(val_env, TrainingConfig.MODEL_SAVE_PATH)
    
    # Treinamento
    logger.info(f"Iniciando treinamento com {TrainingConfig.TOTAL_TIMESTEPS:,} timesteps")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=TrainingConfig.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Treinamento concluído em {training_time}")
        
        # Salva modelo final
        final_model_path = os.path.join(TrainingConfig.MODEL_SAVE_PATH, "final_model")
        model.save(final_model_path)
        logger.info(f"Modelo final salvo em: {final_model_path}")
        
        # Avaliação final
        final_metrics = evaluate_final_model(model, val_env)
        
        # Salva informações do treinamento
        training_info = {
            'training_time': str(training_time),
            'total_timesteps': TrainingConfig.TOTAL_TIMESTEPS,
            'hyperparams': hyperparams,
            'env_kwargs': env_kwargs,
            'final_metrics': final_metrics,
            'data_info': {
                'symbol': TrainingConfig.SYMBOL,
                'interval': TrainingConfig.INTERVAL,
                'n_candles': TrainingConfig.N_CANDLES,
                'train_path': train_path,
                'val_path': val_path
            }
        }
        
        with open(os.path.join(TrainingConfig.MODEL_SAVE_PATH, "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("=== TREINAMENTO CONCLUÍDO COM SUCESSO ===")
        
    except KeyboardInterrupt:
        logger.info("Treinamento interrompido pelo usuário")
        model.save(os.path.join(TrainingConfig.MODEL_SAVE_PATH, "interrupted_model"))
        
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        raise
        
    finally:
        # Cleanup
        train_env.close()
        val_env.close()

if __name__ == "__main__":
    main()
