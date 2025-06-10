"""
Versão Rápida - Otimização com Taxas Reais da Binance
Com medição de tempo e pruning de trials ruins
"""

import optuna
from optuna.exceptions import TrialPruned
import numpy as np
import pandas as pd
import logging
import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import requests
import multiprocessing as mp
from time import time  # ADICIONADO: Para medir tempo de execução

# Primeiro, vamos verificar se os módulos existem
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    print("✅ Stable Baselines 3 importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar Stable Baselines 3: {e}")
    exit(1)

try:
    from scalping_environment import ImprovedScalpingEnv, BinanceFees
    print("✅ Scalping environment importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar scalping_environment: {e}")
    print("Criando classes mock para teste...")
    
    # Classes mock para teste
    class BinanceFees:
        MAKER_FEE = 0.0002  # 0.02%
        TAKER_FEE = 0.0004  # 0.04%
        ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE  # 0.06%
        AVERAGE_FEE = (MAKER_FEE + TAKER_FEE) / 2  # 0.03%
        MIN_PROFIT_THRESHOLD = ROUND_TRIP_FEE * 2  # 0.12%
    
    class ImprovedScalpingEnv:
        def __init__(self, **kwargs):
            print(f"Mock environment criado com: {kwargs}")
            self.observation_space = type('obj', (object,), {'shape': (20,)})()
            self.action_space = type('obj', (object,), {'n': 3})()
        
        def reset(self):
            return np.random.random(20)
        
        def step(self, action):
            return np.random.random(20), np.random.random() * 0.001, False, {}

warnings.filterwarnings("ignore")

# Configuração de logging - WARNING para performance
logging.basicConfig(
    level=logging.WARNING,  # WARNING para menos logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'optuna_fast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizationConfig:
    """Configurações da otimização rápida com foco em taxas reais"""
    
    # Dados
    SYMBOL = "ETHUSDT"
    INTERVAL = "5m"
    N_CANDLES = 5000  # AUMENTADO para mais dados
    
    # Otimização
    N_TRIALS = 25  # AUMENTADO para mais trials
    N_SPLITS = 2
    PARALLEL_ENVS = 2
    TIMESTEPS_PER_MINI_LOOP = 6000  # ALTERADO: Timesteps por mini-loop
    N_MINI_LOOPS = 5  # ALTERADO: Número de mini-loops
    EVAL_STEPS = 750  # AUMENTADO para melhor avaliação
    
    # Early stopping
    PATIENCE = 8  # REDUZIDO para ser mais agressivo
    MIN_IMPROVEMENT = 0.001
    
    # Configurações específicas para taxas
    USE_REALISTIC_FEES = True
    MIN_PROFIT_THRESHOLD = BinanceFees.MIN_PROFIT_THRESHOLD
    MAX_PROFIT_THRESHOLD = 0.01
    
    # Faixa de transaction costs para otimizar
    MIN_TRANSACTION_COST = BinanceFees.MAKER_FEE
    MAX_TRANSACTION_COST = BinanceFees.TAKER_FEE

    # Parallel processing - DESABILITADO para estabilidade
    N_JOBS = 1
    USE_PARALLEL = False

class DataManager:
    """Gerenciador de dados para otimização"""
    
    def __init__(self, symbol: str = "ETHUSDT", interval: str = "5m"):
        self.symbol = symbol
        self.interval = interval
    
    def download_binance_data(self, n_candles: int = 5000) -> str:
        """Baixa dados históricos da Binance"""
        print(f"Baixando {n_candles} candles para {self.symbol}")
        
        try:
            url = "https://api.binance.com/api/v3/klines"
            all_data = []
            limit = 1000  # Limite da API da Binance
            end_time = None
            request_count = 0
            
            # Loop para fazer múltiplos requests até obter n_candles
            while len(all_data) < n_candles:
                request_count += 1
                print(f"Request {request_count}: Solicitando {min(limit, n_candles - len(all_data))} candles...")
                
                params = {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "limit": min(limit, n_candles - len(all_data))
                }
                
                # Adiciona endTime para pegar dados mais antigos
                if end_time:
                    params["endTime"] = end_time
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Se não recebeu dados, para o loop
                if not data:
                    print("Não há mais dados disponíveis")
                    break
                
                # Adiciona os dados no início da lista para manter ordem cronológica
                all_data = data + all_data
                
                # Atualiza o endTime para o próximo request
                end_time = data[0][0] - 1  # timestamp do primeiro candle - 1ms
                
                print(f"Coletados {len(all_data)} candles até agora...")
                
                # Se recebeu menos que o limite, não há mais dados
                if len(data) < limit:
                    break
            
            # Limita ao número solicitado
            all_data = all_data[-n_candles:] if len(all_data) > n_candles else all_data
            
            print(f"Total coletado: {len(all_data)} candles, usando: {len(all_data)} candles")
            
            # Processa dados
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
            
            df = pd.DataFrame(all_data, columns=cols)
            df[["open", "high", "low", "close", "volume"]] = \
                df[["open", "high", "low", "close", "volume"]].astype(float)
            
            # Salva arquivo
            csv_path = f"{self.symbol}_{self.interval}_{len(df)}.csv"
            df.to_csv(csv_path, index=False)
            
            print(f"Dados salvos: {csv_path} ({len(df)} candles)")
            return csv_path
            
        except Exception as e:
            print(f"Erro ao baixar dados: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def create_walk_forward_splits(self, csv_path: str, n_splits: int = 2) -> List[Tuple[str, str]]:
        """Cria splits para walk-forward validation"""
        print(f"Criando {n_splits} splits do arquivo {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
        
        df = pd.read_csv(csv_path)
        total_len = len(df)
        print(f"Dataset tem {total_len} linhas")
        
        # Calcula tamanhos - OTIMIZADO para ser mais rápido
        test_size = max(200, total_len // (n_splits + 2))  # Menor test size
        train_size = max(600, test_size * 2)  # Menor train size
        
        print(f"Train size: {train_size}, Test size: {test_size}")
        
        splits = []
        
        for i in range(n_splits):
            start_idx = i * test_size
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > total_len:
                print(f"Split {i} excede dados disponíveis, pulando")
                break
            
            # Dados de treino
            train_data = df.iloc[start_idx:train_end]
            train_path = f"train_split_{i}.csv"
            train_data.to_csv(train_path, index=False)
            
            # Dados de teste
            test_data = df.iloc[train_end:test_end]
            test_path = f"test_split_{i}.csv"
            test_data.to_csv(test_path, index=False)
            
            splits.append((train_path, test_path))
            
            print(f"Split {i}: Train={len(train_data)}, Test={len(test_data)}")
        
        print(f"Criados {len(splits)} splits")
        return splits

class TradingMetrics:
    """Calculadora de métricas de trading considerando taxas"""
    
    @staticmethod
    def calculate_fee_adjusted_metrics(returns: np.ndarray, transaction_cost: float) -> Dict[str, float]:
        """Calcula métricas ajustadas por taxas"""
        if len(returns) == 0:
            return {
                'gross_return': 0, 'net_return': 0, 'total_fees': 0, 
                'fee_ratio': 1, 'net_sharpe': 0, 'net_win_rate': 0,
                'profit_after_fees': False
            }
        
        # Estima taxas baseado no número de trades
        n_trades = len(returns[returns != 0])
        total_fees = n_trades * transaction_cost * 2  # Round trip
        
        # Métricas brutas
        gross_return = np.sum(returns)
        
        # Métricas líquidas
        net_return = gross_return - total_fees
        net_returns = returns - (transaction_cost * 2 * (returns != 0))
        
        return {
            'gross_return': gross_return,
            'net_return': net_return,
            'total_fees': total_fees,
            'fee_ratio': total_fees / max(abs(gross_return), 1e-8),
            'net_sharpe': TradingMetrics.calculate_sharpe_ratio(net_returns),
            'net_win_rate': len(net_returns[net_returns > 0]) / max(len(net_returns[net_returns != 0]), 1),
            'profit_after_fees': net_return > 0
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calcula Sharpe Ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calcula Maximum Drawdown"""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)

def make_env(csv_path: str, **env_kwargs):
    """Factory para criar ambiente"""
    def _init():
        return ImprovedScalpingEnv(csv_path=csv_path, live_stream=False, **env_kwargs)
    return _init

def evaluate_model(model: PPO, env: DummyVecEnv, n_steps: int = 500) -> Dict[str, float]:
    """Avalia modelo em um ambiente específico e retorna métricas"""
    obs = env.reset()
    episode_returns = []
    
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if reward != 0:  # Só considera rewards não-zero como trades
            episode_returns.append(reward)
        
        if done.any():
            obs = env.reset()
    
    # Calcula métricas
    if not episode_returns:
        return {
            'total_profit': 0.0,
            'average_reward': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
    
    returns_array = np.array(episode_returns)
    total_profit = np.sum(returns_array)
    average_reward = np.mean(returns_array)
    win_rate = len(returns_array[returns_array > 0]) / len(returns_array)
    max_drawdown = TradingMetrics.calculate_max_drawdown(returns_array)
    
    return {
        'total_profit': float(total_profit),
        'average_reward': float(average_reward),
        'win_rate': float(win_rate),
        'max_drawdown': float(max_drawdown)
    }

def objective(trial: optuna.Trial) -> float:
    """
    Função objetivo com medição de tempo e pruning
    """
    # ADICIONADO: Medição de tempo
    start_time = time()
    
    print(f"=== TRIAL {trial.number} ===")
    
    try:
        # SIMPLIFICADO: Hiperparâmetros principais
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        
        # Parâmetros do ambiente - SIMPLIFICADO
        transaction_cost = trial.suggest_float(
            "transaction_cost", 
            OptimizationConfig.MIN_TRANSACTION_COST,
            OptimizationConfig.MAX_TRANSACTION_COST
        )
        
        min_profit = max(OptimizationConfig.MIN_PROFIT_THRESHOLD, transaction_cost * 3)
        profit_threshold = trial.suggest_float(
            "profit_threshold", 
            min_profit,
            OptimizationConfig.MAX_PROFIT_THRESHOLD
        )
        
        stop_loss_threshold = trial.suggest_float("stop_loss_threshold", 0.01, 0.02)
        
        round_trip_cost = transaction_cost * 2
        
        print(f"Trial {trial.number}: LR={learning_rate:.6f}, "
               f"Profit_th={profit_threshold:.4f}, "
               f"Transaction_cost={transaction_cost:.4f}")
        
        # Verificação simples
        if profit_threshold <= round_trip_cost:
            print(f"Trial {trial.number}: Profit threshold inviável")
            return -1.0
        
        # Parâmetros do ambiente
        env_kwargs = {
            'profit_threshold': profit_threshold,
            'stop_loss_threshold': stop_loss_threshold,
            'max_position_time': 40,  # Valor fixo para simplificar
            'transaction_cost': transaction_cost,
            'force_dim': 20,
            'use_realistic_fees': OptimizationConfig.USE_REALISTIC_FEES
        }
        
        # Carrega dados
        data_manager = DataManager(OptimizationConfig.SYMBOL, OptimizationConfig.INTERVAL)
        
        csv_path = f"{OptimizationConfig.SYMBOL}_{OptimizationConfig.INTERVAL}_{OptimizationConfig.N_CANDLES}.csv"
        if not os.path.exists(csv_path):
            print("Baixando dados da Binance...")
            csv_path = data_manager.download_binance_data(OptimizationConfig.N_CANDLES)
        else:
            print(f"Usando dados existentes: {csv_path}")
        
        splits = data_manager.create_walk_forward_splits(csv_path, OptimizationConfig.N_SPLITS)
        
        if not splits:
            print("Nenhum split criado")
            return -1.0
        
        # Treina modelo no último split (mais recente)
        train_path, test_path = splits[-1]
        print(f"Treinando modelo com: {train_path}")
        
        train_env = DummyVecEnv([make_env(train_path, **env_kwargs) 
                                for _ in range(OptimizationConfig.PARALLEL_ENVS)])
        
        test_env = DummyVecEnv([make_env(test_path, **env_kwargs)])
        
        # Cria modelo - SIMPLIFICADO
        print("Criando modelo PPO...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            gamma=0.99,
            batch_size=128,
            n_steps=1024,
            verbose=0,
            device='auto'
        )
        
        # ALTERADO: Treinamento em mini-loops com pruning
        print(f"Treinando em {OptimizationConfig.N_MINI_LOOPS} mini-loops...")
        total_reward = 0
        
        for step in range(1, OptimizationConfig.N_MINI_LOOPS + 1):
            # Treina por um mini-loop
            model.learn(
                total_timesteps=OptimizationConfig.TIMESTEPS_PER_MINI_LOOP, 
                reset_num_timesteps=False
            )
            
            # Avalia após cada mini-loop
            metrics = evaluate_model(
                model, 
                test_env, 
                n_steps=OptimizationConfig.EVAL_STEPS // OptimizationConfig.N_MINI_LOOPS
            )
            
            # Acumula recompensa
            step_reward = metrics['average_reward']
            total_reward += step_reward
            avg_reward = total_reward / step
            
            print(f"Mini-loop {step}/{OptimizationConfig.N_MINI_LOOPS}: "
                  f"Reward={step_reward:.6f}, Avg={avg_reward:.6f}")
            
            # Reporta para o pruner
            trial.report(avg_reward, step)
            
            # Verifica se deve fazer pruning
            if trial.should_prune():
                print(f"Trial {trial.number} pruned após {step} mini-loops")
                raise TrialPruned()
        
        # Avaliação final completa
        final_metrics = evaluate_model(model, test_env, n_steps=OptimizationConfig.EVAL_STEPS)
        
        # ADICIONADO: Salva métricas detalhadas no trial
        trial.set_user_attr("total_profit", final_metrics['total_profit'])
        trial.set_user_attr("average_reward", final_metrics['average_reward'])
        trial.set_user_attr("win_rate", final_metrics['win_rate'])
        trial.set_user_attr("max_drawdown", final_metrics['max_drawdown'])
        trial.set_user_attr("train_time_sec", round(time() - start_time, 2))
        
        print(f"Trial {trial.number} - Métricas finais: "
              f"Profit={final_metrics['total_profit']:.6f}, "
              f"Avg Reward={final_metrics['average_reward']:.6f}, "
              f"Win Rate={final_metrics['win_rate']:.2f}, "
              f"Tempo={trial.user_attrs['train_time_sec']}s")
        
        # Cleanup
        train_env.close()
        test_env.close()
        
        # Limpa arquivos temporários
        for train_path, test_path in splits:
            try:
                os.remove(train_path)
                os.remove(test_path)
            except:
                pass
        
        # ALTERADO: Retorna average_reward como objetivo
        return final_metrics['average_reward']
        
    except TrialPruned:
        # Repropaga a exceção para o Optuna
        raise
    except Exception as e:
        print(f"ERRO no trial {trial.number}: {e}")
        import traceback
        print(traceback.format_exc())
        return -1.0

def main():
    """Função principal de otimização"""
    print("=== OTIMIZAÇÃO RÁPIDA COM TAXAS REAIS DA BINANCE ===")
    print(f"Configurações:")
    print(f"  - N_TRIALS: {OptimizationConfig.N_TRIALS}")
    print(f"  - N_SPLITS: {OptimizationConfig.N_SPLITS}")
    print(f"  - MINI_LOOPS: {OptimizationConfig.N_MINI_LOOPS} x {OptimizationConfig.TIMESTEPS_PER_MINI_LOOP} timesteps")
    print(f"  - EVAL_STEPS: {OptimizationConfig.EVAL_STEPS}")
    
    # Cria estudo com pruner
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1
        )
    )
    
    try:
        print("Iniciando otimização...")
        study.optimize(objective, n_trials=OptimizationConfig.N_TRIALS)
        
    except KeyboardInterrupt:
        print("Otimização interrompida pelo usuário")
    except Exception as e:
        print(f"Erro na otimização: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Resultados
    if study.trials:
        print("=== RESULTADOS ===")
        print(f"Trials completados: {len(study.trials)}")
        print(f"Trials pruned: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
        
        if study.best_trial:
            print(f"Melhores hiperparâmetros: {study.best_params}")
            print(f"Melhor score: {study.best_value:.6f}")
            
            # Mostra métricas detalhadas do melhor trial
            best_trial = study.best_trial
            print("Métricas do melhor trial:")
            print(f"  - Total Profit: {best_trial.user_attrs.get('total_profit', 'N/A')}")
            print(f"  - Average Reward: {best_trial.user_attrs.get('average_reward', 'N/A')}")
            print(f"  - Win Rate: {best_trial.user_attrs.get('win_rate', 'N/A'):.2f}")
            print(f"  - Max Drawdown: {best_trial.user_attrs.get('max_drawdown', 'N/A')}")
            print(f"  - Training Time: {best_trial.user_attrs.get('train_time_sec', 'N/A')}s")
        
        # Salva resultados
        os.makedirs("optimization_results", exist_ok=True)
        
        with open("optimization_results/best_hyperparams_fast.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
        
        # Salva métricas detalhadas
        if study.best_trial:
            metrics = {k: v for k, v in study.best_trial.user_attrs.items()}
            with open("optimization_results/best_metrics_fast.json", "w") as f:
                json.dump(metrics, f, indent=2)
        
        # Salva todos os trials
        try:
            study.trials_dataframe().to_csv("optimization_results/all_trials_fast.csv", index=False)
        except:
            print("Erro ao salvar trials_dataframe")
        
        print("Resultados salvos em optimization_results/")
    else:
        print("Nenhum trial foi completado!")
    
    return study.best_params if study.trials and study.best_trial else {}

if __name__ == "__main__":
    print("Iniciando otimização rápida...")
    best_params = main()
    print(f"Concluído! Melhores parâmetros: {best_params}")
