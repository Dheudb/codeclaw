import os
from dotenv import load_dotenv
from loguru import logger

import sys
# Make sure we can import codeclaw and bot directly from e:\LLM\claude-code-main\codeclaw
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codeclaw.core.config import load_config, save_config, load_config_into_env
from bot.feishu.gateway import FeishuGateway
from rich.console import Console

console = Console()

def interactive_setup(cfg: dict) -> dict:
    updated = False
    console.print("\n[bold cyan]🚀 欢迎使用 CodeClaw Feishu Bot！[/bold cyan]")
    console.print("🔧 检测到缺少必要配置，我们将带您完成初始化：\n")
    
    if not cfg.get("feishu_app_id"):
        cfg["feishu_app_id"] = console.input("[yellow]请输入 Feishu API App ID: [/yellow]").strip()
        updated = True
    if not cfg.get("feishu_app_secret"):
        cfg["feishu_app_secret"] = console.input("[yellow]请输入 Feishu API App Secret: [/yellow]").strip()
        updated = True
        
    if updated:
        save_config(cfg)
        console.print("\n[bold green]✅ 配置已保存到 ~/.codeclaw/config.json 全局储存中！[/bold green]\n")
        
    return cfg

def main():
    # 将 ~/.codeclaw/config.json 中的模型配置注入环境变量
    # 这样后续创建的 QueryEngine 能自动读取 API Key / Base URL / Model
    load_config_into_env()
    
    cfg = load_config()
    
    # 交互式检查并填补配置
    if not cfg.get("feishu_app_id") or not cfg.get("feishu_app_secret"):
        cfg = interactive_setup(cfg)
        
    app_id = cfg.get("feishu_app_id")
    app_secret = cfg.get("feishu_app_secret")
    
    if not app_id or not app_secret:
        logger.error("Please set FEISHU_APP_ID and FEISHU_APP_SECRET in .env")
        return
        
    gateway = FeishuGateway(
        app_id=app_id,
        app_secret=app_secret,
        encrypt_key=cfg.get("feishu_encrypt_key", ""),
        verification_token=cfg.get("feishu_verification_token", "")
    )
    
    try:
        gateway.start()
    except KeyboardInterrupt:
        logger.info("Bot shutting down...")

if __name__ == "__main__":
    main()
