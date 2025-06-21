import importlib.util
import os
import sys

class Plugin:
    def __init__(self, app_data, plugin_settings=None):
        self.name = "viewer"
        self.settings = plugin_settings or {}
        # UIクラスをonly_content.pyから動的にimport
        self._load_ui_class()
    
    def _load_ui_class(self):
        """only_content.pyからUIクラスを動的にimportする"""
        # only_content.pyのパスを取得
        current_dir = os.path.dirname(__file__)
        only_content_path = os.path.join(current_dir, "only_content.py")
        
        # モジュールの読み込み
        spec = importlib.util.spec_from_file_location("only_content", only_content_path)
        only_content_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(only_content_module)
        
        # UIクラスを取得してインスタンス作成
        UI = only_content_module.UI
        self.ui = UI()
    
    def on_flag(self, app_data):
        """フラグ処理メイン - AI、ツール、ユーザーの表示を適切に管理"""
        # AI応答後に表示（ツール呼び出しやAIメッセージを含む）
        if app_data["flags"]["after_ai_response"] and not app_data["flags"]["ready_for_user"]:
            # only_content.pyのUIクラスを使用してメッセージを表示
            self.ui.print(app_data.messages)
            # ready_for_userフラグを設定して、次のユーザー入力を受け付けられるようにする
            app_data["flags"]["ready_for_user"] = True
    
    def cleanup(self):
        """クリーンアップ処理"""
        pass