from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

def display(messages):
    """メッセージのcontentのみを表示する関数"""
    console = Console()
    
    # 表示するメッセージを絞り込む
    display_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            # システムメッセージは表示しない
            continue
        display_messages.append(message)
    
    # 最新の会話のみを表示（最大2つ：最新のユーザーメッセージとAIの応答）
    if len(display_messages) > 2:
        display_messages = display_messages[-2:]
    
    for message in display_messages:
        if isinstance(message, HumanMessage):
            # ユーザーメッセージ
            text = Text(f"ユーザー: {message.content}")
            text.stylize("bold blue")
            console.print(text)
        
        elif isinstance(message, AIMessage):
            # AIメッセージ
            if isinstance(message.content, list):
                # contentがリストの場合（ツール使用など）
                for item in message.content:
                    if isinstance(item, dict) and 'text' in item:
                        text = Text(f"AI: {item['text']}")
                        text.stylize("bold green")
                        console.print(text)
            else:
                # 通常のテキスト
                text = Text(f"AI: {message.content}")
                text.stylize("bold green")
                console.print(text)
        
        elif isinstance(message, ToolMessage):
            # ツールメッセージ
            console.print(f"---")
            console.print(message.content)
        
        else:
            # その他のメッセージタイプ
            console.print(f"その他: {message.content}")
        
        console.print("---")


