from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

def display(messages):
    """メッセージのcontentを適切に表示する関数 - AI、ツール、ユーザー、システムを明確に区別"""
    console = Console()
    
    # 全てのメッセージを表示対象とする（SystemMessageも含む）
    display_messages = messages
    
    # 最新のユーザーメッセージから始まる一連の会話を表示
    if len(display_messages) > 0:
        # 最後のHumanMessageのインデックスを見つける
        last_human_idx = -1
        for i in range(len(display_messages) - 1, -1, -1):
            if isinstance(display_messages[i], HumanMessage):
                last_human_idx = i
                break
        
        if last_human_idx >= 0:
            # SystemMessageがある場合は、それも含めて表示
            # SystemMessageは通常会話の最初にあるので、さらに前を確認
            start_idx = last_human_idx
            for i in range(last_human_idx - 1, -1, -1):
                if isinstance(display_messages[i], SystemMessage):
                    start_idx = i
                    break
                elif isinstance(display_messages[i], HumanMessage):
                    # 別のHumanMessageが見つかったら停止
                    break
            
            display_messages = display_messages[start_idx:]
    
    # メッセージタイプごとに適切に表示
    for message in display_messages:
        if isinstance(message, SystemMessage):
            # システムメッセージ
            system_text = Text(f"システム: {message.content}")
            system_text.stylize("bold cyan")
            console.print(system_text)
        
        elif isinstance(message, HumanMessage):
            # ユーザーメッセージ
            user_text = Text(f"ユーザー: {message.content}")
            user_text.stylize("bold blue")
            console.print(user_text)
        
        elif isinstance(message, AIMessage):
            # AIメッセージ
            ai_content = _extract_ai_content(message)
            
            # AIのコンテンツがある場合のみ表示
            if ai_content.strip():
                ai_text = Text(f"AI: {ai_content}")
                ai_text.stylize("bold green")
                console.print(ai_text)
            
            # ツール呼び出しがある場合は、それも表示
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown_tool')
                    tool_text = Text(f"AI → ツール呼び出し: {tool_name}")
                    tool_text.stylize("bold yellow")
                    console.print(tool_text)
        
        elif isinstance(message, ToolMessage):
            # ツールメッセージ - ツール実行結果を表示
            tool_text = Text(f"ツール: {message.content}")
            tool_text.stylize("bold magenta")
            console.print(tool_text)
        
        else:
            # その他のメッセージタイプ（念のため）
            other_text = Text(f"その他: {message.content}")
            other_text.stylize("dim")
            console.print(other_text)
        
        # メッセージ間の区切り線
        console.print("---")
    
    # 同期処理のため、出力を確実にフラッシュ
    import sys
    sys.stdout.flush()
    sys.stderr.flush()


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
        return str(content)


