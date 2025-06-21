from rich.console import Console
from rich.text import Text
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import sys

def display(messages):
    """メッセージのcontentを適切に表示する関数 - シンプルな順次表示"""
    print(f"[DEBUG] display() called with {len(messages)} messages", file=sys.stderr)
    console = Console()
    
    # 全てのメッセージを順番に表示
    for i, message in enumerate(messages):
        print(f"[DEBUG] Processing message {i+1}: {type(message).__name__}", file=sys.stderr)
        
        if isinstance(message, SystemMessage):
            # システムメッセージ
            system_text = Text(f"システム: {message.content}")
            system_text.stylize("bold cyan")
            console.print(system_text)
            print(f"[DEBUG] Displayed SystemMessage", file=sys.stderr)
        
        elif isinstance(message, HumanMessage):
            # ユーザーメッセージ
            user_text = Text(f"ユーザー: {message.content}")
            user_text.stylize("bold blue")
            console.print(user_text)
            print(f"[DEBUG] Displayed HumanMessage", file=sys.stderr)
        
        elif isinstance(message, AIMessage):
            # AIメッセージ
            ai_content = _extract_ai_content(message)
            
            # AIのコンテンツがある場合のみ表示
            if ai_content and ai_content.strip():
                ai_text = Text(f"AI: {ai_content}")
                ai_text.stylize("bold green")
                console.print(ai_text)
                print(f"[DEBUG] Displayed AIMessage with content", file=sys.stderr)
            
            # ツール呼び出しがある場合は、それも表示
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown_tool')
                    tool_text = Text(f"AI → ツール呼び出し: {tool_name}")
                    tool_text.stylize("bold yellow")
                    console.print(tool_text)
                    print(f"[DEBUG] Displayed tool call: {tool_name}", file=sys.stderr)
        
        elif isinstance(message, ToolMessage):
            # ツールメッセージ - ツール実行結果を表示
            tool_text = Text(f"ツール: {message.content}")
            tool_text.stylize("bold magenta")
            console.print(tool_text)
            print(f"[DEBUG] Displayed ToolMessage", file=sys.stderr)
        
        # メッセージ間の区切り線
        console.print("---")
    
    print(f"[DEBUG] About to flush stdout/stderr", file=sys.stderr)
    # 表示完了を確実にするため、コンソールをフラッシュ
    sys.stdout.flush()
    sys.stderr.flush()
    print(f"[DEBUG] Flush completed", file=sys.stderr)


def _extract_ai_content(ai_message: AIMessage) -> str:
    """AIMessageからコンテンツを抽出する"""
    content = ai_message.content
    
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # contentがリストの場合（Anthropicなど）
        extracted_content = ""
        for item in content:
            if isinstance(item, dict):
                if 'text' in item:
                    extracted_content += item['text']
                elif 'content' in item:
                    extracted_content += str(item['content'])
            elif isinstance(item, str):
                extracted_content += item
        return extracted_content
    else:
        # その他の形式の場合は文字列に変換
        return str(content) if content is not None else ""


