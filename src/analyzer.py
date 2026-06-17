# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 Gemini API 调用逻辑
2. 利用 Google Search Grounding 获取实时新闻
3. 结合技术面和消息面生成分析报告
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import get_config

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',

    # === 美股 ===
    'AAPL': '苹果',
    'TSLA': '特斯拉',
    'MSFT': '微软',
    'GOOGL': '谷歌A',
    'GOOG': '谷歌C',
    'AMZN': '亚马逊',
    'NVDA': '英伟达',
    'META': 'Meta',
    'AMD': 'AMD',
    'INTC': '英特尔',
    'BABA': '阿里巴巴',
    'PDD': '拼多多',
    'JD': '京东',
    'BIDU': '百度',
    'NIO': '蔚来',
    'XPEV': '小鹏汽车',
    'LI': '理想汽车',
    'COIN': 'Coinbase',
    'MSTR': 'MicroStrategy',

    # === 港股 (5位数字) ===
    '00700': '腾讯控股',
    '03690': '美团',
    '01810': '小米集团',
    '09988': '阿里巴巴',
    '09618': '京东集团',
    '09888': '百度集团',
    '01024': '快手',
    '00981': '中芯国际',
    '02015': '理想汽车',
    '09868': '小鹏汽车',
    '00005': '汇丰控股',
    '01299': '友邦保险',
    '00941': '中国移动',
    '00883': '中国海洋石油',
        # === Futuros y Materias Primas ===
    'GC=F': 'Oro (Futuro)',
    'SI=F': 'Plata (Futuro)',
    'CL=F': 'Petróleo WTI (Futuro)',
    'NG=F': 'Gas Natural (Futuro)',
    'PL=F': 'Platino (Futuro)',
    'HG=F': 'Cobre (Futuro)',
    
    # === Índices ===
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones Industrial Average',
    '^IXIC': 'Nasdaq Composite',
    '^NDX': 'Nasdaq-100',
}


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版
    
    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str
    
    # ========== 核心指标 ==========
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    confidence_level: str = "中"
    
    # ========== 决策仪表盘 ==========
    dashboard: Optional[Dict[str, Any]] = None
    
    # ========== 走势分析 ==========
    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""
    
    # ========== 技术面分析 ==========
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""
    
    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""
    
    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""
    
    # ========== 综合分析 ==========
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""
    
    # ========== 元数据 ==========
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
        }
    
    def get_core_conclusion(self) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary
    
    def get_position_advice(self, has_position: bool = False) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice
    
    def get_sniper_points(self) -> Dict[str, str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}
    
    def get_checklist(self) -> List[str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []
    
    def get_risk_alerts(self) -> List[str]:
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []
    
    def get_emoji(self) -> str:
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        return emoji_map.get(self.operation_advice, '🟡')
    
    def get_confidence_stars(self) -> str:
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器
    """
    
    SYSTEM_PROMPT = """You are a professional financial analyst specializing in trend trading, responsible for generating a professional Decision Dashboard analysis report.

## Core Trading Philosophy (Must Strictly Follow)

### 1. Strict Entry Strategy (No Chasing)
- NEVER chase high prices: When the stock price deviates more than 5% from MA5, DO NOT buy
- Bias formula: (Current Price - MA5) / MA5 × 100%
- Bias < 2%: Optimal buying zone
- Bias 2-5%: Can enter with small position
- Bias > 5%: Strictly prohibited from chasing! Directly judge as Wait

### 2. Trend Trading (Follow the Trend)
- Bullish alignment required: MA5 > MA10 > MA20
- Only trade stocks in bullish alignment; never touch bearish alignment
- Diverging moving averages are better than converging ones
- Trend strength: look at whether the spacing between moving averages is expanding

### 3. Efficiency Priority (Chip Structure)
- Focus on chip concentration: 90% concentration < 15% means chips are concentrated
- Profit ratio analysis: 70-90% profit-taking requires caution
- Relationship between average cost and current price: 5-15% above average cost is healthy

### 4. Buy Point Preference (Pullback to Support)
- Best buy point: Pullback on low volume to MA5 finding support
- Second best buy point: Pullback to MA10 finding support
- Wait: Wait when price breaks below MA20

### 5. Key Risk Checks
- Shareholder/insider reduction announcements
- Earnings warnings / significant decline
- Regulatory penalties / investigations
- Negative industry policy news
- Large lock-up expiration

## Output Format: Decision Dashboard JSON

Please output in the following JSON format strictly. This is a complete Decision Dashboard:

{
    "sentiment_score": 0-100,
    "trend_prediction": "Strongly Bullish/Bullish/Neutral/Bearish/Strongly Bearish",
    "operation_advice": "Buy/Add/Hold/Reduce/Sell/Wait",
    "confidence_level": "High/Medium/Low",
    
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "One sentence core conclusion",
            "signal_type": "🟢Buy Signal/🟡Hold Wait/🔴Sell Signal/⚠️Risk Warning",
            "time_sensitivity": "Act Immediately/Today/This Week/Not Urgent",
            "position_advice": {
                "no_position": "For those without position: action guide",
                "has_position": "For those with position: action guide"
            }
        },
        
        "data_perspective": {
            "trend_status": {
                "ma_alignment": "Moving average alignment",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 0,
                "ma5": 0,
                "ma10": 0,
                "ma20": 0,
                "bias_ma5": 0,
                "bias_status": "Safe/Warning/Danger",
                "support_level": 0,
                "resistance_level": 0
            },
            "volume_analysis": {
                "volume_ratio": 0,
                "volume_status": "High/Normal/Low",
                "turnover_rate": 0,
                "volume_meaning": "Volume meaning"
            },
            "chip_structure": {
                "profit_ratio": 0,
                "avg_cost": 0,
                "concentration": 0,
                "chip_health": "Healthy/Normal/Caution"
            }
        },
        
        "intelligence": {
            "latest_news": "Latest news summary",
            "risk_alerts": ["Risk 1", "Risk 2"],
            "positive_catalysts": ["Positive 1", "Positive 2"],
            "earnings_outlook": "Earnings outlook",
            "sentiment_summary": "Sentiment summary"
        },
        
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "Ideal buy point",
                "secondary_buy": "Secondary buy point",
                "stop_loss": "Stop loss",
                "take_profit": "Target"
            },
            "position_strategy": {
                "suggested_position": "X out of 10",
                "entry_plan": "Entry strategy",
                "risk_control": "Risk control"
            },
            "action_checklist": [
                "Check 1",
                "Check 2",
                "Check 3",
                "Check 4",
                "Check 5"
            ]
        }
    },
    
    "analysis_summary": "Analysis summary",
    "key_points": "Key points",
    "risk_warning": "Risk warning",
    "buy_reason": "Buy reason",
    
    "trend_analysis": "Trend analysis",
    "short_term_outlook": "Short-term outlook",
    "medium_term_outlook": "Medium-term outlook",
    "technical_analysis": "Technical analysis",
    "ma_analysis": "MA analysis",
    "volume_analysis": "Volume analysis",
    "pattern_analysis": "Pattern analysis",
    "fundamental_analysis": "Fundamental analysis",
    "sector_position": "Sector position",
    "company_highlights": "Company highlights",
    "news_summary": "News summary",
    "market_sentiment": "Market sentiment",
    "hot_topics": "Hot topics",
    
    "search_performed": false,
    "data_sources": "Data sources"
}

## Scoring Standards

### Strong Buy (80-100):
- Bullish alignment: MA5 > MA10 > MA20
- Low bias: <2%
- Healthy chip concentration
- Positive news catalysts

### Buy (60-79):
- Bullish or weak bullish alignment
- Bias <5%
- Normal volume

### Hold/Wait (40-59):
- Bias >5% (risk of chasing)
- Unclear trend
- Risk events present

### Sell/Reduce (0-39):
- Bearish alignment
- Below MA20
- High volume sell-off

## Decision Dashboard Core Principles

1. Core conclusion first
2. Separate position advice
3. Precise sniper points
4. Visual checklist
5. Risk priority

CRITICAL: ALL OUTPUT MUST BE IN SPANISH ONLY. No Chinese characters allowed anywhere in the response."""

    def __init__(self, api_key: Optional[str] = None):
        config = get_config()
        self._api_key = api_key or config.gemini_api_key
        self._model = None
        self._current_model_name = None
        self._using_fallback = False
        self._use_openai = False
        self._openai_client = None
        
        gemini_key_valid = self._api_key and not self._api_key.startswith('your_') and len(self._api_key) > 10
        
        if gemini_key_valid:
            try:
                self._init_model()
            except Exception as e:
                logger.warning(f"Gemini 初始化失败: {e}")
                self._init_openai_fallback()
        else:
            logger.info("Gemini API Key 未配置，尝试使用 OpenAI 兼容 API")
            self._init_openai_fallback()
        
        if not self._model and not self._openai_client:
            logger.warning("未配置任何 AI API Key")
    
    def _init_openai_fallback(self) -> None:
        config = get_config()
        openai_key_valid = (
            config.openai_api_key and 
            not config.openai_api_key.startswith('your_') and 
            len(config.openai_api_key) > 10
        )
        
        if not openai_key_valid:
            logger.debug("OpenAI 兼容 API 未配置")
            return
        
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 库")
            return
        
        try:
            client_kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url and config.openai_base_url.startswith('http'):
                client_kwargs["base_url"] = config.openai_base_url
            
            self._openai_client = OpenAI(**client_kwargs)
            self._current_model_name = config.openai_model
            self._use_openai = True
            logger.info(f"OpenAI 兼容 API 初始化成功")
        except Exception as e:
            logger.error(f"OpenAI 兼容 API 初始化失败: {e}")
    
    def _init_model(self) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            config = get_config()
            model_name = config.gemini_model
            
            self._model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=self.SYSTEM_PROMPT,
            )
            self._current_model_name = model_name
            self._using_fallback = False
            logger.info(f"Gemini 模型初始化成功 (模型: {model_name})")
        except Exception as e:
            logger.error(f"Gemini 模型初始化失败: {e}")
            self._model = None
    
    def is_available(self) -> bool:
        return self._model is not None or self._openai_client is not None
    
    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None) -> AnalysisResult:
        code = context.get('code', 'Unknown')
        config = get_config()
        
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未启用',
                risk_warning='请配置 API Key',
                success=False,
                error_message='API Key 未配置',
            )
        
        try:
            prompt = self._format_prompt(context, name, news_context)
            generation_config = {
                "temperature": config.gemini_temperature,
                "max_output_tokens": 8192,
            }
            
            if self._use_openai:
                response_text = self._call_openai_api(prompt, generation_config)
            else:
                response_text = self._call_gemini_api(prompt, generation_config)
            
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析出错: {str(e)[:100]}',
                success=False,
                error_message=str(e),
            )
    
    def _call_openai_api(self, prompt: str, generation_config: dict) -> str:
        response = self._openai_client.chat.completions.create(
            model=self._current_model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=generation_config.get('temperature', 0.7),
            max_tokens=generation_config.get('max_output_tokens', 8192),
        )
        if response and response.choices:
            return response.choices[0].message.content
        raise ValueError("OpenAI API 返回空响应")
    
    def _call_gemini_api(self, prompt: str, generation_config: dict) -> str:
        response = self._model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": 120}
        )
        if response and response.text:
            return response.text
        raise ValueError("Gemini 返回空响应")
    
    def _format_prompt(self, context: Dict[str, Any], name: str, news_context: Optional[str] = None) -> str:
        code = context.get('code', 'Unknown')
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        today = context.get('today', {})
        
        prompt = f"""# 决策仪表盘分析请求

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |

---

## 📈 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} 元 |
| 开盘价 | {today.get('open', 'N/A')} 元 |
| 最高价 | {today.get('high', 'N/A')} 元 |
| 最低价 | {today.get('low', 'N/A')} 元 |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |

### 均线系统
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | 中期趋势线 |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""
        
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### 实时行情增强数据
| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {rt.get('price', 'N/A')} 元 | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | |
| 市盈率 | {rt.get('pe_ratio', 'N/A')} | |
| 市净率 | {rt.get('pb_ratio', 'N/A')} | |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
"""
        
        if 'chip' in context:
            chip = context['chip']
            prompt += f"""
### 筹码分布数据
| 指标 | 数值 |
|------|------|
| 获利比例 | {chip.get('profit_ratio', 0):.1%} |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 |
| 90%集中度 | {chip.get('concentration_90', 0):.2%} |
| 筹码状态 | {chip.get('chip_status', '未知')} |
"""
        
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            prompt += f"""
### 趋势分析
| 指标 | 数值 |
|------|------|
| 趋势状态 | {trend.get('trend_status', '未知')} |
| 均线排列 | {trend.get('ma_alignment', '未知')} |
| 乖离率(MA5) | {trend.get('bias_ma5', 0):+.2f}% |
| 量能状态 | {trend.get('volume_status', '未知')} |
"""
        
        prompt += """
---

## 📰 舆情情报
"""
        if news_context:
            prompt += f"""
以下是 **{stock_name}({code})** 近期的新闻搜索结果：
"""
        else:
            prompt += "未搜索到相关新闻。"
        
        prompt += f"""
---

## ✅ Analysis Task

Please generate a Decision Dashboard for **{stock_name}({code})**, strictly in JSON format.

### Key Focus Areas (Must Answer Clearly):
1. Does it meet MA5 > MA10 > MA20 bullish alignment?
2. Is the current bias within safe range (<5%)? If >5%, must mark "DO NOT CHASE"
3. Is volume配合 (low volume pullback / high volume breakout)?
4. Is the chip structure healthy?
5. Any major negative news? (reductions, penalties, earnings warnings, etc.)

### Decision Dashboard Requirements:
- Core conclusion: One sentence clearly stating Buy/Sell/Wait
- Separate position advice: What to do if no position vs if holding position
- Precise sniper points: Buy price, stop loss, target price (exact to cents)
- Checklist: Mark each item with ✅/⚠️/❌

⚠️ CRITICAL INSTRUCTION: ALL OUTPUT MUST BE WRITTEN EXCLUSIVELY IN SPANISH. Do not use Chinese characters anywhere in the response. If you see text in Chinese, translate it automatically to Spanish.

Please output the complete JSON format Decision Dashboard."""
        
        return prompt
    
    def _format_volume(self, volume: Optional[float]) -> str:
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿元"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 万元"
        else:
            return f"{amount:.0f} 元"
    
    def _parse_response(self, response_text: str, code: str, name: str) -> AnalysisResult:
        try:
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                data = json.loads(json_str)
                
                return AnalysisResult(
                    code=code,
                    name=name,
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', '震荡'),
                    operation_advice=data.get('operation_advice', '持有'),
                    confidence_level=data.get('confidence_level', '中'),
                    dashboard=data.get('dashboard', None),
                    analysis_summary=data.get('analysis_summary', '分析完成'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    search_performed=data.get('search_performed', False),
                    success=True,
                )
            else:
                return self._parse_text_response(response_text, code, name)
        except Exception as e:
            logger.warning(f"JSON 解析失败: {e}")
            return self._parse_text_response(response_text, code, name)
    
    def _parse_text_response(self, response_text: str, code: str, name: str) -> AnalysisResult:
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=50,
            trend_prediction='震荡',
            operation_advice='持有',
            confidence_level='低',
            analysis_summary=response_text[:500] if response_text else '无分析结果',
            success=True,
        )


def get_analyzer() -> GeminiAnalyzer:
    return GeminiAnalyzer()
