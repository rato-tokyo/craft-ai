import sys
import time
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

def display_module(messages):
    """メッセージを受け取り、画面に表示するべき文字列を返す関数"""
    output_lines = []
    
    # 全てのメッセージを処理
    for i, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            # システムメッセージ（シアン色、太字）
            output_lines.append(f"\033[1;36mシステム: {message.content}\033[0m")
        
        elif isinstance(message, HumanMessage):
            # ユーザーメッセージ（青色、太字）
            output_lines.append(f"\033[1;34mユーザー: {message.content}\033[0m")
        
        elif isinstance(message, AIMessage):
            # AIメッセージ
            ai_content = _extract_ai_content(message)
            
            # AIのコンテンツがある場合のみ表示（緑色、太字）
            if ai_content and ai_content.strip():
                output_lines.append(f"\033[1;32mAI: {ai_content}\033[0m")
            
            # ツール呼び出しがある場合は、それも表示（黄色、太字）
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown_tool')
                    output_lines.append(f"\033[1;33mAI → ツール呼び出し: {tool_name}\033[0m")
        
        elif isinstance(message, ToolMessage):
            # ToolMessage - ツール実行結果を表示（マゼンタ色、太字）
            output_lines.append(f"\033[1;35mツール: {message.content}\033[0m")
        
        # 最後のメッセージ以外に区切り線を追加
        if i < len(messages) - 1:
            output_lines.append("---")
    
    return "\n".join(output_lines)


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

