from binance.client import Client
from binance.enums import *
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os
from stable_baselines3 import PPO
from scalping_environment import ImprovedScalpingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES - EDITE AQUI
# ============================================================================

# Credenciais do Testnet
TESTNET_API_KEY = "7a1cbcb710e41a5999ce0e01a7cf22666066849fed712f5ddd77bb130684ce70"
TESTNET_SECRET_KEY = "e08378154a7a2e76f47daff2b448bf871802c4527da21a344e2f95846de94449"

# Configurações de Trading
TRADING_CONFIG = {
    "symbol": "ETHUSDT",
    "quantity": 0.01,  # Quantidade por trade
    "stop_loss_percent": 0.02,  # 2%
    "take_profit_percent": 0.03,  # 3%
    "sleep_time": 30,  # Segundos entre verificações
    "use_testnet": True,  # True = testnet, False = produção
    "model_path": "trained_models/final_model.zip",  # Caminho relativo do modelo PPO
    "lookback_periods": 20  # Períodos para análise técnica
}

# ============================================================================

class PPOFuturesBot:
    """Bot usando modelo PPO treinado para decisões de trading"""
    
    def __init__(self):
        # Configurações
        self.symbol = TRADING_CONFIG["symbol"]
        self.quantity = TRADING_CONFIG["quantity"]
        self.stop_loss_percent = TRADING_CONFIG["stop_loss_percent"]
        self.take_profit_percent = TRADING_CONFIG["take_profit_percent"]
        self.sleep_time = TRADING_CONFIG["sleep_time"]
        self.use_testnet = TRADING_CONFIG["use_testnet"]
        self.model_path = TRADING_CONFIG["model_path"]
        self.lookback_periods = TRADING_CONFIG["lookback_periods"]
        
        # Inicializa cliente
        if self.use_testnet:
            self.client = Client(
                api_key=TESTNET_API_KEY,
                api_secret=TESTNET_SECRET_KEY,
                testnet=True
            )
            logger.info("🧪 Conectado ao Testnet da Binance Futures")
        else:
            logger.error("❌ Credenciais de produção não configuradas")
            raise ValueError("Configure suas credenciais de produção")
        
        # Carrega modelo PPO
        self.model = None
        self.load_ppo_model()
        
        # Dados históricos para análise
        self.price_history = []
        self.volume_history = []
        
        # Estatísticas
        self.stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "start_time": datetime.now(),
            "model_predictions": 0,
            "random_actions": 0
        }
    
    def load_ppo_model(self):
        """Carrega o modelo PPO treinado"""
        try:
            if os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path)
                logger.info(f"✅ Modelo PPO carregado: {self.model_path}")
                logger.info("🤖 Bot usará modelo treinado para decisões")
            else:
                logger.warning(f"⚠️  Modelo não encontrado: {self.model_path}")
                logger.info("🎲 Bot usará estratégia aleatória como fallback")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            logger.info("🎲 Bot usará estratégia aleatória como fallback")
    
    def test_connection(self):
        """Testa conexão com a API"""
        try:
            server_time = self.client.futures_time()['serverTime']
            account = self.client.futures_account(recvWindow=5000, timestamp=server_time)
            logger.info("✅ Conexão estabelecida com sucesso")
            
            balance = float(account['totalWalletBalance'])
            logger.info(f"💰 Saldo total: {balance} USDT")
            return True
        except Exception as e:
            logger.error(f"❌ Erro na conexão: {e}")
            return False
    
    def get_current_price(self):
        """Obtém preço atual"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Erro ao obter preço: {e}")
            return 0.0
    
    def get_market_data(self):
        """Obtém dados de mercado mais completos"""
        try:
            # Preço atual
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Estatísticas 24h
            stats = self.client.futures_ticker(symbol=self.symbol)
            volume = float(stats['volume'])
            price_change = float(stats['priceChangePercent'])
            
            # Klines recentes para análise técnica
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=self.lookback_periods
            )
            
            # Processa klines
            prices = [float(k[4]) for k in klines]  # Close prices
            volumes = [float(k[5]) for k in klines]  # Volumes
            
            return {
                "current_price": current_price,
                "volume_24h": volume,
                "price_change_24h": price_change,
                "recent_prices": prices,
                "recent_volumes": volumes
            }
        except Exception as e:
            logger.error(f"Erro ao obter dados de mercado: {e}")
            return None
    
    def prepare_observation(self, market_data):
        """Prepara observação para o modelo PPO"""
        try:
            if not market_data or len(market_data["recent_prices"]) < self.lookback_periods:
                return None
            
            prices = np.array(market_data["recent_prices"])
            volumes = np.array(market_data["recent_volumes"])
            
            # Calcula indicadores técnicos simples
            returns = np.diff(prices) / prices[:-1]
            
            # RSI simples
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # Médias móveis
            sma_5 = np.mean(prices[-5:])
            sma_10 = np.mean(prices[-10:])
            
            # Volume médio
            avg_volume = np.mean(volumes)
            
            # Monta observação (deve ter 20 features para o modelo)
            observation = np.array([
                # Preços normalizados (últimos 10)
                *((prices[-10:] - prices[-10]) / (prices[-10] + 1e-8)),
                # Returns (últimos 5)
                *(returns[-5:] if len(returns) >= 5 else np.zeros(5)),
                # Indicadores
                rsi / 100.0,  # RSI normalizado
                (sma_5 - sma_10) / (sma_10 + 1e-8),  # Diferença SMA
                market_data["price_change_24h"] / 100.0,  # Mudança 24h
                avg_volume / (avg_volume + 1e-8),  # Volume normalizado
                # Padding para completar 20 features
                0.0, 0.0
            ])
            
            # Garante que tem exatamente 20 features
            if len(observation) > 20:
                observation = observation[:20]
            elif len(observation) < 20:
                observation = np.pad(observation, (0, 20 - len(observation)))
            
            return observation.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erro ao preparar observação: {e}")
            return None
    
    def get_trading_signal(self, market_data):
        """Obtém sinal de trading usando modelo PPO ou fallback"""
        if self.model and market_data:
            try:
                # Prepara observação
                obs = self.prepare_observation(market_data)
                if obs is not None:
                    # Predição do modelo
                    action, _ = self.model.predict(obs, deterministic=True)
                    self.stats["model_predictions"] += 1
                    
                    # Converte ação numérica para string
                    if action == 0:
                        return "HOLD"
                    elif action == 1:
                        return "LONG"
                    elif action == 2:
                        return "SHORT"
                    else:
                        return "HOLD"
            except Exception as e:
                logger.error(f"Erro na predição do modelo: {e}")
        
        # Fallback: estratégia aleatória
        import random
        self.stats["random_actions"] += 1
        action = random.choices([0, 1, 2], weights=[80, 10, 10])[0]
        
        if action == 1:
            return "LONG"
        elif action == 2:
            return "SHORT"
        else:
            return "HOLD"
    
    def get_position(self):
        """Obtém posição atual"""
        try:
            server_time = self.client.futures_time()['serverTime']
            positions = self.client.futures_position_information(symbol=self.symbol, recvWindow=5000, timestamp=server_time)
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    amount = float(pos['positionAmt'])
                    if amount != 0:
                        return {
                            "side": "LONG" if amount > 0 else "SHORT",
                            "amount": abs(amount),
                            "entry_price": float(pos['entryPrice']),
                            "pnl": float(pos['unRealizedProfit'])
                        }
            return None
        except Exception as e:
            logger.error(f"Erro ao obter posição: {e}")
            return None
    
    def open_position(self, side):
        """Abre posição (LONG ou SHORT)"""
        try:
            order_side = SIDE_BUY if side == "LONG" else SIDE_SELL
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=self.quantity
            )
            
            logger.info(f"📈 Posição {side} aberta: {self.quantity} {self.symbol}")
            logger.info(f"   Order ID: {order['orderId']}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao abrir posição {side}: {e}")
            return False
    
    def close_position(self, position):
        """Fecha posição atual"""
        try:
            order_side = SIDE_SELL if position["side"] == "LONG" else SIDE_BUY
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=position["amount"]
            )
            
            # Atualiza estatísticas
            self.stats["trades"] += 1
            self.stats["total_pnl"] += position["pnl"]
            
            if position["pnl"] > 0:
                self.stats["wins"] += 1
                logger.info(f"🎯 Posição {position['side']} fechada com LUCRO: {position['pnl']:.4f} USDT")
            else:
                self.stats["losses"] += 1
                logger.info(f"⚠️ Posição {position['side']} fechada com PREJUÍZO: {position['pnl']:.4f} USDT")
            
            logger.info(f"   Order ID: {order['orderId']}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição: {e}")
            return False
    
    def check_stop_loss_take_profit(self, position, current_price):
        """Verifica se deve fechar por stop loss ou take profit"""
        entry_price = position["entry_price"]
        
        if position["side"] == "LONG":
            if current_price <= entry_price * (1 - self.stop_loss_percent):
                logger.info(f"🛑 STOP LOSS ativado para LONG")
                return True
            if current_price >= entry_price * (1 + self.take_profit_percent):
                logger.info(f"🎯 TAKE PROFIT ativado para LONG")
                return True
        elif position["side"] == "SHORT":
            if current_price >= entry_price * (1 + self.stop_loss_percent):
                logger.info(f"🛑 STOP LOSS ativado para SHORT")
                return True
            if current_price <= entry_price * (1 - self.take_profit_percent):
                logger.info(f"🎯 TAKE PROFIT ativado para SHORT")
                return True
        
        return False
    
    def print_stats(self):
        """Imprime estatísticas do bot"""
        win_rate = (self.stats["wins"] / max(self.stats["trades"], 1)) * 100
        runtime = datetime.now() - self.stats["start_time"]
        
        model_usage = self.stats["model_predictions"] / max(
            self.stats["model_predictions"] + self.stats["random_actions"], 1
        ) * 100
        
        logger.info("=" * 60)
        logger.info("📊 ESTATÍSTICAS DO BOT PPO")
        logger.info(f"⏱️  Tempo de execução: {runtime}")
        logger.info(f"📈 Total de trades: {self.stats['trades']}")
        logger.info(f"🏆 Trades vencedores: {self.stats['wins']}")
        logger.info(f"📉 Trades perdedores: {self.stats['losses']}")
        logger.info(f"🎯 Taxa de acerto: {win_rate:.1f}%")
        logger.info(f"💰 PnL total: {self.stats['total_pnl']:.4f} USDT")
        logger.info(f"🤖 Uso do modelo PPO: {model_usage:.1f}%")
        logger.info(f"🎲 Ações aleatórias: {self.stats['random_actions']}")
        logger.info("=" * 60)
    
    def run(self):
        """Executa o bot principal"""
        logger.info("🚀 INICIANDO BOT DE FUTURES COM MODELO PPO")
        logger.info(f"📊 Símbolo: {self.symbol}")
        logger.info(f"💰 Quantidade por trade: {self.quantity}")
        logger.info(f"🛡️ Stop Loss: {self.stop_loss_percent:.1%}")
        logger.info(f"🎯 Take Profit: {self.take_profit_percent:.1%}")
        logger.info(f"🤖 Modelo: {'PPO Carregado' if self.model else 'Estratégia Aleatória'}")
        
        # Testa conexão
        if not self.test_connection():
            logger.error("❌ Falha na conexão. Encerrando...")
            return
        
        try:
            while True:
                # Obtém dados de mercado
                market_data = self.get_market_data()
                if not market_data:
                    logger.warning("⚠️ Dados de mercado inválidos, pulando ciclo...")
                    time.sleep(self.sleep_time)
                    continue
                
                current_price = market_data["current_price"]
                
                # Verifica posição atual
                position = self.get_position()
                
                if position:
                    # Tem posição aberta - verifica stop loss/take profit
                    logger.info(f"📍 Posição ativa: {position['side']} | "
                              f"Quantidade: {position['amount']} | "
                              f"PnL: {position['pnl']:.4f} USDT")
                    
                    if self.check_stop_loss_take_profit(position, current_price):
                        self.close_position(position)
                else:
                    # Sem posição - obtém sinal do modelo
                    signal = self.get_trading_signal(market_data)
                    
                    model_indicator = "🤖" if self.model else "🎲"
                    logger.info(f"📊 Preço: {current_price:.2f} USDT | "
                              f"Sinal: {signal} {model_indicator} | "
                              f"Mudança 24h: {market_data['price_change_24h']:.2f}%")
                    
                    if signal in ["LONG", "SHORT"]:
                        self.open_position(signal)
                
                # Mostra estatísticas a cada 5 trades
                if self.stats["trades"] > 0 and self.stats["trades"] % 5 == 0:
                    self.print_stats()
                
                # Aguarda próximo ciclo
                time.sleep(self.sleep_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Bot interrompido pelo usuário")
            
            # Fecha posições abertas
            position = self.get_position()
            if position:
                logger.info("🔄 Fechando posições abertas...")
                self.close_position(position)
        
        except Exception as e:
            logger.error(f"❌ Erro fatal: {e}")
        
        finally:
            self.print_stats()
            logger.info("👋 Bot encerrado")

if __name__ == "__main__":
    print("🤖 BOT DE FUTURES BINANCE COM MODELO PPO")
    print("=" * 60)
    print("✅ Usando biblioteca python-binance padrão")
    print("✅ Testnet configurado")
    print("✅ Modelo PPO integrado")
    print("✅ Stop Loss e Take Profit automáticos")
    print("✅ Análise técnica em tempo real")
    print()
    
    bot = PPOFuturesBot()
    bot.run()
