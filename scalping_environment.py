"""
Ambiente de Scalping Corrigido com Taxas Reais da Binance
Incorpora taxas de transação realistas no treinamento
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import requests
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BinanceFees:
    """Configurações de taxas da Binance"""
    
    # Taxas Binance Futures (padrão)
    MAKER_FEE = 0.0002  # 0.02%
    TAKER_FEE = 0.0004  # 0.04%
    
    # Taxa média (assumindo mix de maker/taker)
    AVERAGE_FEE = (MAKER_FEE + TAKER_FEE) / 2  # 0.03%
    
    # Taxa total por operação completa (entrada + saída)
    ROUND_TRIP_FEE = AVERAGE_FEE * 2  # 0.06%
    
    # Profit threshold mínimo (deve ser > que round trip fee)
    MIN_PROFIT_THRESHOLD = ROUND_TRIP_FEE * 1.5  # 0.09% (50% acima das taxas)
    
    @classmethod
    def get_realistic_profit_threshold(cls) -> float:
        """Retorna profit threshold realista considerando taxas"""
        # Para scalping, queremos pelo menos 2-3x as taxas como profit mínimo
        return cls.ROUND_TRIP_FEE * 3  # 0.18% (3x as taxas)

class StreamingOHLCV:
    """Classe para streaming de dados OHLCV da Binance"""
    
    def __init__(self, symbol: str = "ETHUSDT", interval: str = "5m", ohlcv_len: int = 150):
        self.symbol = symbol.upper()
        self.interval = interval
        self.ohlcv_len = ohlcv_len
        self.last_data = None
        
    def get_latest(self, interval: str = None, n: int = 150) -> pd.DataFrame:
        """Obtém dados mais recentes da Binance API"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": n
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                if self.last_data is not None:
                    return self.last_data
                raise ValueError("Nenhum dado retornado da API")
            
            cols = ["open_time", "open", "high", "low", "close", "volume",
                   "close_time", "quote_asset_volume", "number_of_trades",
                   "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
            
            df = pd.DataFrame(data, columns=cols)
            df[["open", "high", "low", "close", "volume"]] = \
                df[["open", "high", "low", "close", "volume"]].astype(float)
            
            # Cache dos dados
            self.last_data = df[["open", "high", "low", "close", "volume"]].copy()
            return self.last_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados da API: {e}")
            if self.last_data is not None:
                return self.last_data
            # Retorna dados dummy em caso de falha total
            return self._generate_dummy_data(n)
    
    def _generate_dummy_data(self, n: int) -> pd.DataFrame:
        """Gera dados dummy para fallback"""
        np.random.seed(42)
        base_price = 2000
        # Aumenta volatilidade para permitir profits acima das taxas
        prices = base_price + np.cumsum(np.random.normal(0, 8, n))  # Mais volatilidade
        
        return pd.DataFrame({
            'open': prices + np.random.normal(0, 3, n),
            'high': prices + np.abs(np.random.normal(0, 15, n)),  # Mais spread
            'low': prices - np.abs(np.random.normal(0, 15, n)),
            'close': prices + np.random.normal(0, 3, n),
            'volume': np.random.uniform(100, 500, n)
        })

class CSVStreamingOHLCV:
    """Streaming de dados a partir de arquivo CSV"""
    
    def __init__(self, csv_path: str, ohlcv_len: int = 150):
        self.df = pd.read_csv(csv_path)
        self.ohlcv_len = ohlcv_len
        self.pointer = ohlcv_len
        
        # Valida colunas necessárias
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"CSV deve conter colunas: {required_cols}")
    
    def get_latest(self, interval: str = None, n: int = 150) -> pd.DataFrame:
        """Retorna janela de dados do CSV"""
        if self.pointer >= len(self.df):
            self.pointer = self.ohlcv_len  # Reinicia do início
        
        start_idx = max(0, self.pointer - n)
        data = self.df.iloc[start_idx:self.pointer][['open', 'high', 'low', 'close', 'volume']].copy()
        self.pointer += 1
        
        return data
    
    def reset(self):
        """Reinicia o ponteiro"""
        self.pointer = self.ohlcv_len

class ImprovedScalpingEnv(gym.Env):
    """
    Ambiente de Scalping com Taxas Reais da Binance
    Incorpora custos de transação realistas no treinamento
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        symbol: str = "ETHUSDT",
        force_dim: int = 20,
        profit_threshold: Optional[float] = None,  # Será calculado automaticamente se None
        stop_loss_threshold: float = 0.02,
        max_position_time: int = 50,
        ohlcv_len: int = 150,
        transaction_cost: Optional[float] = None,  # Será definido como taxa real da Binance se None
        live_stream: bool = False,
        csv_path: Optional[str] = None,
        use_realistic_fees: bool = True  # Novo parâmetro para usar taxas reais
    ):
        super().__init__()
        
        self.symbol = symbol
        self.force_dim = force_dim
        self.stop_loss_threshold = stop_loss_threshold
        self.max_position_time = max_position_time
        self.use_realistic_fees = use_realistic_fees
        
        # Configura taxas realistas da Binance
        if use_realistic_fees:
            self.transaction_cost = transaction_cost or BinanceFees.AVERAGE_FEE
            self.profit_threshold = profit_threshold or BinanceFees.get_realistic_profit_threshold()
            logger.info(f"Usando taxas realistas da Binance:")
            logger.info(f"  - Transaction cost: {self.transaction_cost:.4f} ({self.transaction_cost*100:.2f}%)")
            logger.info(f"  - Profit threshold: {self.profit_threshold:.4f} ({self.profit_threshold*100:.2f}%)")
            logger.info(f"  - Round trip cost: {BinanceFees.ROUND_TRIP_FEE:.4f} ({BinanceFees.ROUND_TRIP_FEE*100:.2f}%)")
        else:
            self.transaction_cost = transaction_cost or 0.0004
            self.profit_threshold = profit_threshold or 0.004
        
        # Indicadores técnicos
        self.indicators = [
            "close", "volume", "ema9", "ema21", "ema50", "rsi", "macd", "macd_signal",
            "bb_mavg", "bb_high", "bb_low", "atr", "obv", "price_change", "volume_change"
        ]
        
        # Configuração do streaming de dados
        if csv_path:
            self.streaming = CSVStreamingOHLCV(csv_path, ohlcv_len)
        elif live_stream:
            self.streaming = StreamingOHLCV(symbol, ohlcv_len=ohlcv_len)
        else:
            # Fallback para dados dummy
            self.streaming = StreamingOHLCV(symbol, ohlcv_len=ohlcv_len)
        
        # Estado do ambiente
        self.reset_state()
        
        # Espaços de ação e observação
        self.action_space = gym.spaces.Discrete(3)  # 0=Hold, 1=Long, 2=Short
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.force_dim,), dtype=np.float32
        )
        
        # Carrega dados iniciais
        self.data = self._load_and_process_data()
        
        # Estatísticas de taxas (para debugging)
        self.total_fees_paid = 0.0
        self.total_gross_profit = 0.0
        self.total_net_profit = 0.0
    
    def reset_state(self):
        """Reinicia estado do ambiente"""
        self.position = 0  # 0=sem posição, 1=long, -1=short
        self.entry_price = 0.0
        self.position_steps = 0
        self.total_reward = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.equity_curve = [1.0]  # Começa com equity normalizado
        
        # Reset estatísticas de taxas
        self.total_fees_paid = 0.0
        self.total_gross_profit = 0.0
        self.total_net_profit = 0.0
    
    def _load_and_process_data(self) -> pd.DataFrame:
        """Carrega e processa dados com indicadores técnicos"""
        try:
            df = self.streaming.get_latest(n=150)
            
            if df.empty or len(df) < 55:
                logger.warning("Dados insuficientes, usando dados dummy")
                df = self._generate_fallback_data()
            
            # Calcula indicadores técnicos
            df = self._calculate_indicators(df)
            
            # Remove NaN e retorna apenas indicadores selecionados
            df_clean = df.dropna().reset_index(drop=True)
            
            if len(df_clean) == 0:
                logger.error("Todos os dados são NaN após cálculo de indicadores")
                df_clean = self._generate_fallback_data()
                df_clean = self._calculate_indicators(df_clean).dropna().reset_index(drop=True)
            
            return df_clean[self.indicators] if len(df_clean) > 0 else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Erro ao processar dados: {e}")
            return self._generate_fallback_data()
    
    def _generate_fallback_data(self) -> pd.DataFrame:
        """Gera dados de fallback com volatilidade adequada para superar taxas"""
        np.random.seed(42)
        n = 150
        base_price = 2000
        
        # Volatilidade maior para permitir profits acima das taxas da Binance
        volatility = 15  # Aumentado para permitir movimentos > 0.18%
        prices = base_price + np.cumsum(np.random.normal(0, volatility, n))
        
        return pd.DataFrame({
            'open': prices + np.random.normal(0, 5, n),
            'high': prices + np.abs(np.random.normal(0, 20, n)),  # Spread maior
            'low': prices - np.abs(np.random.normal(0, 20, n)),
            'close': prices + np.random.normal(0, 5, n),
            'volume': np.random.uniform(100, 500, n)
        })
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos os indicadores técnicos"""
        try:
            # EMAs
            df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
            df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
            df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            
            # RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_mavg'] = bb.bollinger_mavg()
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            
            # ATR
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['atr'] = atr.average_true_range()
            
            # OBV
            obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            df['obv'] = obv.on_balance_volume()
            
            # Indicadores adicionais
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            # Retorna DataFrame com indicadores zerados
            for indicator in self.indicators:
                if indicator not in df.columns:
                    df[indicator] = 0.0
            return df
    
    def _get_trend_signal(self) -> int:
        """Determina sinal de tendência baseado em múltiplos indicadores"""
        if self.data.empty:
            return 0
        
        try:
            latest = self.data.iloc[-1]
            
            # Condições para tendência de alta
            bullish_conditions = [
                latest['close'] > latest['ema9'] > latest['ema21'],  # Alinhamento de EMAs
                latest['rsi'] > 50 and latest['rsi'] < 80,  # RSI em zona favorável
                latest['macd'] > latest['macd_signal'],  # MACD positivo
                latest['close'] > latest['bb_mavg'],  # Acima da média das Bollinger
                latest['price_change'] > 0  # Movimento de preço positivo
            ]
            
            # Condições para tendência de baixa
            bearish_conditions = [
                latest['close'] < latest['ema9'] < latest['ema21'],  # Alinhamento de EMAs
                latest['rsi'] < 50 and latest['rsi'] > 20,  # RSI em zona favorável
                latest['macd'] < latest['macd_signal'],  # MACD negativo
                latest['close'] < latest['bb_mavg'],  # Abaixo da média das Bollinger
                latest['price_change'] < 0  # Movimento de preço negativo
            ]
            
            bullish_score = sum(bullish_conditions)
            bearish_score = sum(bearish_conditions)
            
            if bullish_score >= 3:
                return 1  # Tendência de alta
            elif bearish_score >= 3:
                return -1  # Tendência de baixa
            else:
                return 0  # Sem tendência clara
                
        except Exception as e:
            logger.error(f"Erro ao calcular tendência: {e}")
            return 0
    
    def _get_observation(self) -> np.ndarray:
        """Constrói observação do estado atual"""
        if self.data.empty:
            return np.zeros(self.force_dim, dtype=np.float32)
        
        try:
            # Dados dos indicadores
            obs_data = self.data.iloc[-1].values.astype(np.float32)
            
            # Adiciona informações de estado (incluindo custos)
            state_info = np.array([
                self.position,  # Posição atual
                self.position_steps / self.max_position_time,  # Tempo normalizado em posição
                self.total_reward,  # Recompensa acumulada
                self.transaction_cost,  # Custo de transação atual
                self.profit_threshold,  # Threshold de profit atual
            ], dtype=np.float32)
            
            # Combina observações
            full_obs = np.concatenate([obs_data, state_info])
            
            # Normalização robusta
            obs_mean = np.mean(full_obs)
            obs_std = np.std(full_obs)
            if obs_std > 1e-8:
                full_obs = (full_obs - obs_mean) / obs_std
            
            # Ajusta para dimensão fixa
            if len(full_obs) < self.force_dim:
                full_obs = np.concatenate([full_obs, np.zeros(self.force_dim - len(full_obs))])
            elif len(full_obs) > self.force_dim:
                full_obs = full_obs[:self.force_dim]
            
            return full_obs.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erro ao construir observação: {e}")
            return np.zeros(self.force_dim, dtype=np.float32)
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """Calcula recompensa incorporando taxas reais da Binance"""
        reward = 0.0
        
        try:
            # Componente 1: Recompensa por ações alinhadas com tendência
            trend = self._get_trend_signal()
            if (trend == 1 and action == 1) or (trend == -1 and action == 2):
                reward += 0.001  # Pequeno bônus por seguir tendência
            elif (trend == 1 and action == 2) or (trend == -1 and action == 1):
                reward -= 0.002  # Penalização por ir contra tendência
            
            # Componente 2: Recompensa baseada em posição
            if self.position != 0:
                # Calcula P&L não realizado
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if self.position == -1:  # Short position
                    pnl_pct = -pnl_pct
                
                # Recompensa contínua baseada no P&L (descontando taxas estimadas)
                net_pnl = pnl_pct - self.transaction_cost  # Desconta taxa de entrada
                reward += net_pnl * 0.1
                
                # Penalização por tempo em posição (evita hold infinito)
                reward -= 0.0001 * self.position_steps
                
                # Bônus por atingir profit threshold (já considera taxas)
                if pnl_pct >= self.profit_threshold:
                    reward += 0.01
                
                # Penalização severa por stop loss
                if pnl_pct <= -self.stop_loss_threshold:
                    reward -= 0.05
            
            # Componente 3: Recompensa por fechar posições (incorpora taxas reais)
            if action == 0 and self.position != 0:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if self.position == -1:
                    pnl_pct = -pnl_pct
                
                # Calcula P&L líquido após taxas
                gross_pnl = pnl_pct
                total_fees = self.transaction_cost * 2  # Entrada + saída
                net_pnl = gross_pnl - total_fees
                
                # Recompensa baseada no P&L líquido
                reward += net_pnl * 10  # Amplifica recompensa de fechamento
                
                # Atualiza estatísticas de taxas
                self.total_fees_paid += total_fees
                self.total_gross_profit += gross_pnl
                self.total_net_profit += net_pnl
                
                # Log para debugging (ocasional)
                if self.trade_count % 10 == 0:
                    logger.debug(f"Trade {self.trade_count}: Gross PnL: {gross_pnl:.4f}, "
                               f"Fees: {total_fees:.4f}, Net PnL: {net_pnl:.4f}")
            
            # Componente 4: Recompensa por abrir posições
            elif (action == 1 or action == 2) and self.position == 0:
                reward += 0.0005  # Pequeno incentivo para tomar posições
                # Não desconta taxa aqui, será descontada no fechamento
            
            return reward
            
        except Exception as e:
            logger.error(f"Erro ao calcular recompensa: {e}")
            return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Executa um passo no ambiente"""
        try:
            # Obtém preço atual
            current_price = self.data.iloc[-1]['close'] if not self.data.empty else 2000.0
            
            # Calcula recompensa
            reward = self._calculate_reward(action, current_price)
            
            # Executa ação
            done = False
            info = {}
            
            if action == 1 and self.position == 0:  # Abrir Long
                self.position = 1
                self.entry_price = current_price
                self.position_steps = 0
                info['action'] = 'OPEN_LONG'
                
            elif action == 2 and self.position == 0:  # Abrir Short
                self.position = -1
                self.entry_price = current_price
                self.position_steps = 0
                info['action'] = 'OPEN_SHORT'
                
            elif action == 0 and self.position != 0:  # Fechar posição
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if self.position == -1:
                    pnl_pct = -pnl_pct
                
                # Calcula P&L líquido após taxas
                net_pnl = pnl_pct - (self.transaction_cost * 2)
                
                self.trade_count += 1
                if net_pnl > 0:  # Considera apenas lucro líquido
                    self.winning_trades += 1
                
                info['action'] = 'CLOSE_POSITION'
                info['gross_pnl_pct'] = pnl_pct
                info['net_pnl_pct'] = net_pnl
                info['fees_paid'] = self.transaction_cost * 2
                info['trade_duration'] = self.position_steps
                
                self.position = 0
                self.entry_price = 0.0
                self.position_steps = 0
            
            # Atualiza contador de steps em posição
            if self.position != 0:
                self.position_steps += 1
                
                # Stop loss automático
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                if self.position == -1:
                    pnl_pct = -pnl_pct
                
                if pnl_pct <= -self.stop_loss_threshold:
                    reward -= 0.05  # Penalização adicional
                    net_pnl = pnl_pct - (self.transaction_cost * 2)
                    
                    self.position = 0
                    self.entry_price = 0.0
                    self.position_steps = 0
                    info['action'] = 'STOP_LOSS'
                    info['gross_pnl_pct'] = pnl_pct
                    info['net_pnl_pct'] = net_pnl
                    info['fees_paid'] = self.transaction_cost * 2
                
                # Força fechamento após muito tempo em posição
                elif self.position_steps >= self.max_position_time:
                    reward -= 0.01  # Penalização por hold excessivo
                    net_pnl = pnl_pct - (self.transaction_cost * 2)
                    
                    self.position = 0
                    self.entry_price = 0.0
                    self.position_steps = 0
                    info['action'] = 'FORCED_CLOSE'
                    info['gross_pnl_pct'] = pnl_pct
                    info['net_pnl_pct'] = net_pnl
                    info['fees_paid'] = self.transaction_cost * 2
            
            # Atualiza dados para próximo step
            self.data = self._load_and_process_data()
            
            # Atualiza estatísticas
            self.total_reward += reward
            self.equity_curve.append(self.equity_curve[-1] + reward)
            
            # Calcula drawdown
            peak = max(self.equity_curve)
            current_dd = (peak - self.equity_curve[-1]) / peak if peak > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_dd)
            
            # Informações adicionais
            info.update({
                'position': self.position,
                'position_steps': self.position_steps,
                'total_reward': self.total_reward,
                'trade_count': self.trade_count,
                'win_rate': self.winning_trades / max(self.trade_count, 1),
                'max_drawdown': self.max_drawdown,
                'current_price': current_price,
                'total_fees_paid': self.total_fees_paid,
                'total_net_profit': self.total_net_profit,
                'profit_threshold': self.profit_threshold,
                'transaction_cost': self.transaction_cost
            })
            
            obs = self._get_observation()
            
            return obs, reward, done, False, info
            
        except Exception as e:
            logger.error(f"Erro no step: {e}")
            return self._get_observation(), 0.0, False, False, {'error': str(e)}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reinicia o ambiente"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reinicia streaming se disponível
        if hasattr(self.streaming, 'reset'):
            self.streaming.reset()
        
        # Reinicia estado
        self.reset_state()
        
        # Recarrega dados
        self.data = self._load_and_process_data()
        
        obs = self._get_observation()
        info = {
            'reset': True,
            'data_length': len(self.data),
            'profit_threshold': self.profit_threshold,
            'transaction_cost': self.transaction_cost,
            'binance_round_trip_fee': BinanceFees.ROUND_TRIP_FEE
        }
        
        return obs, info
    
    def render(self, mode: str = 'human'):
        """Renderização do ambiente (placeholder)"""
        pass
    
    def close(self):
        """Fecha o ambiente"""
        pass
    
    def get_fee_statistics(self) -> dict:
        """Retorna estatísticas de taxas para análise"""
        return {
            'total_fees_paid': self.total_fees_paid,
            'total_gross_profit': self.total_gross_profit,
            'total_net_profit': self.total_net_profit,
            'fee_ratio': self.total_fees_paid / max(abs(self.total_gross_profit), 1e-8),
            'profit_threshold': self.profit_threshold,
            'transaction_cost': self.transaction_cost,
            'binance_round_trip_fee': BinanceFees.ROUND_TRIP_FEE
        }

# Função de conveniência para criar ambiente
def make_scalping_env(**kwargs):
    """Factory function para criar ambiente de scalping"""
    return ImprovedScalpingEnv(**kwargs)

if __name__ == "__main__":
    # Teste básico do ambiente com taxas reais
    print("=== TESTE DO AMBIENTE COM TAXAS REAIS DA BINANCE ===")
    
    env = ImprovedScalpingEnv(live_stream=False, use_realistic_fees=True)
    obs, info = env.reset()
    
    print(f"Configuração de taxas:")
    print(f"  - Profit threshold: {env.profit_threshold:.4f} ({env.profit_threshold*100:.2f}%)")
    print(f"  - Transaction cost: {env.transaction_cost:.4f} ({env.transaction_cost*100:.2f}%)")
    print(f"  - Binance round trip: {BinanceFees.ROUND_TRIP_FEE:.4f} ({BinanceFees.ROUND_TRIP_FEE*100:.2f}%)")
    
    print(f"\nObservação inicial: {obs.shape}")
    print(f"Info: {info}")
    
    # Testa alguns steps
    total_reward = 0
    for i in range(20):
        action = np.random.randint(0, 3)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if info.get('action') and info['action'] != 'HOLD':
            print(f"Step {i}: Action={action}, Reward={reward:.6f}, "
                  f"Info={info.get('action')}, "
                  f"Net PnL={info.get('net_pnl_pct', 0):.4f}")
    
    # Estatísticas finais
    fee_stats = env.get_fee_statistics()
    print(f"\n=== ESTATÍSTICAS DE TAXAS ===")
    for key, value in fee_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nReward total: {total_reward:.6f}")
    env.close()
