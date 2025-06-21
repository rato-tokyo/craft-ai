from rich.console import Console
import importlib.util
import os
import sys

class Plugin:
    def __init__(self, app_data, plugin_settings=None):
        self.name = "viewer"
        self.settings = plugin_settings or {}
    
    def on_flag(self, app_data):
        """フラグ処理メイン - AI、ツール、ユーザーの表示を適切に管理"""
        # AI応答後に表示（ツール呼び出しやAIメッセージを含む）
        if app_data["flags"]["after_ai_response"] and not app_data["flags"]["ready_for_user"]:
            # AppDataのUIクラスを使用してメッセージを表示
            app_data.ui.print(app_data.messages)
            # ready_for_userフラグを設定して、次のユーザー入力を受け付けられるようにする
            app_data["flags"]["ready_for_user"] = True
    
    def cleanup(self):
        """クリーンアップ処理"""
        pass