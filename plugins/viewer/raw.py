from rich.console import Console

def display(messages):
    """メッセージをそのままprintする関数"""
    console = Console()
    console.print("===== メッセージ一覧（生データ） =====")
    
    for i, message in enumerate(messages):
        console.print(f"[bold cyan]メッセージ {i+1}:[/bold cyan]")
        console.print(message)
        console.print("---")
    
    console.print("===== メッセージ一覧（終了） =====") 