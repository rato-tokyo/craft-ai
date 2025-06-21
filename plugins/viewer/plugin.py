from rich.console import Console
import importlib.util
import os
import sys

class Plugin:
    def __init__(self, app_data, plugin_settings=None):
        self.name = "viewer"
        self.console = Console()
        self.settings = plugin_settings or {}
        self.display_module = None
        
        # 表示モジュールの読み込み
        if "display_module" in self.settings and self.settings["display_module"]:
            self._load_display_module(self.settings["display_module"])
    
    def _load_display_module(self, module_path):
        """表示モジュールを動的に読み込む"""
        if not os.path.exists(module_path):
            pass  # エラーは無視
            return
            
        # モジュール名を取得
        module_name = os.path.basename(module_path).replace('.py', '')
        
        # モジュールを読み込む
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        self.display_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.display_module)
    
    def on_flag(self, app_data):
        """フラグ処理メイン - AI、ツール、ユーザーの表示を適切に管理"""
        # 設定が変更された場合は再読み込み
        if "display_module" in self.settings and self.settings["display_module"] and not self.display_module:
            self._load_display_module(self.settings["display_module"])
        
        # AI応答後に表示（ツール呼び出しやAIメッセージを含む）
        if app_data["flags"]["after_ai_response"] and not app_data["flags"]["ready_for_user"]:
            # AI応答とツール実行結果を表示
            self._display_messages(app_data.messages)
            # ready_for_userフラグを設定して、次のユーザー入力を受け付けられるようにする
            app_data["flags"]["ready_for_user"] = True
    
    def _display_messages(self, messages):
        """メッセージを表示する（同期処理）"""
        if self.display_module and hasattr(self.display_module, "display"):
            self.display_module.display(messages)
            # 表示完了を確実にするため、コンソールをフラッシュ
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
        else:
            # デフォルトの表示方法は無効化
            pass
            # 表示完了を確実にするため、コンソールをフラッシュ
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
    
    def cleanup(self):
        """クリーンアップ処理"""
        pass