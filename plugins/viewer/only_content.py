import sys
import time
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

class UI:
    """統一UI管理クラス"""
    
    def __init__(self):
        """UIクラスの初期化"""
        # Windows環境での画面クリア対応
        self.is_windows = os.name == 'nt'
    
    def clear_screen(self):
        """画面クリア（Windows対応）"""
        print('\033[2J\033[H', end='')
        sys.stdout.flush()
        time.sleep(0.5)
        return 
    
    def print(self, messages):
        """メッセージを表示する統一メソッド"""
        
        # 画面クリア
        self.clear_screen()
            
        
        # 全てのメッセージを表示
        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                # システムメッセージ（シアン色、太字）
                print(f"\033[1;36mシステム: {message.content}\033[0m")
            
            elif isinstance(message, HumanMessage):
                # ユーザーメッセージ（青色、太字）
                print(f"\033[1;34mユーザー: {message.content}\033[0m")
            
            elif isinstance(message, AIMessage):
                # AIメッセージ
                ai_content = self._extract_ai_content(message)
                
                # AIのコンテンツがある場合のみ表示（緑色、太字）
                if ai_content and ai_content.strip():
                    print(f"\033[1;32mAI: {ai_content}\033[0m")
                
                # ツール呼び出しがある場合は、それも表示（黄色、太字）
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.get('name', 'unknown_tool')
                        print(f"\033[1;33mAI → ツール呼び出し: {tool_name}\033[0m")
            
            elif isinstance(message, ToolMessage):
                # ToolMessage - ツール実行結果を表示（マゼンタ色、太字）
                print(f"\033[1;35mツール: {message.content}\033[0m")
            
            # 最後のメッセージ以外に区切り線を追加
            if i < len(messages) - 1:
                print("---")
        
        
    
    def _extract_ai_content(self, ai_message: AIMessage) -> str:
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


def display(messages, ui=None):
    """メッセージのcontentをUI.print経由で表示する関数"""
    # UIインスタンスが提供されない場合は新しく作成
    if ui is None:
        ui = UI()
    # UI.printを使用して表示
    ui.print(messages)


