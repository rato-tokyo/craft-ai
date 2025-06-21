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
            self.console.print(f"[bold red]エラー: 表示モジュール {module_path} が見つかりません[/bold red]")
            return
            
        try:
            # モジュール名を取得
            module_name = os.path.basename(module_path).replace('.py', '')
            
            # モジュールを読み込む
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.display_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.display_module)
        except Exception as e:
            self.console.print(f"[bold red]エラー: 表示モジュールの読み込みに失敗しました: {str(e)}[/bold red]")
            self.display_module = None
    
    def on_flag(self, app_data):
        """フラグ処理メイン"""
        # 設定が変更された場合は再読み込み
        if "display_module" in self.settings and self.settings["display_module"] and not self.display_module:
            self._load_display_module(self.settings["display_module"])
            
        # ユーザー入力後かつAI準備完了時に表示
        if app_data["flags"]["after_user_input"] and app_data["flags"]["ready_for_ai"]:
            # AI送信前の監視
            self._display_messages(app_data.messages)
            # ready_for_userフラグを設定して、AI応答後の表示を有効にする
            app_data["flags"]["ready_for_user"] = True
        
        # AI応答後のみ表示
        elif app_data["flags"]["after_ai_response"] and not app_data["flags"]["ready_for_user"]:
            # AI受信後の監視
            self._display_messages(app_data.messages)
            # ready_for_userフラグを設定して、次のユーザー入力を受け付けられるようにする
            app_data["flags"]["ready_for_user"] = True
    
    def _display_messages(self, messages):
        """メッセージを表示する"""
        if self.display_module and hasattr(self.display_module, "display"):
            self.display_module.display(messages)
        else:
            # デフォルトの表示方法
            self.console.print(messages)
    
    def cleanup(self):
        """クリーンアップ処理"""
        pass