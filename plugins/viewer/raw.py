import sys
import time
import os
import ast
import pprint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

def display_module(messages):
    """メッセージを受け取り、画面に表示するべき文字列を返す関数"""
    return format_messages(str(messages))

import re
def format_messages(messages_str: str) -> str:
    """
    メッセージ履歴をシンプルにフォーマットする
    
    Args:
        messages_str: メッセージ履歴の文字列表現
    
    Returns:
        str: フォーマット済みのメッセージ履歴
    """
    # SystemMessage, HumanMessage, AIMessage, ToolMessage を改行で区切る
    formatted = re.sub(r'(SystemMessage|HumanMessage|AIMessage|ToolMessage)\(', r'\n\1(\n  ', messages_str)
    # カンマの後に改行を追加
    formatted = re.sub(r', (?=[a-z_]+=)', r',\n  ', formatted)
    # 閉じ括弧の前に改行
    formatted = re.sub(r'\), ', r'\n),\n', formatted)
    return formatted.strip()