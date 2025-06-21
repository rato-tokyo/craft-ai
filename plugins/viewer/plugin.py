import importlib.util
import os
import sys
import time

class Plugin:
    def __init__(self, app_data, plugin_settings=None):
        self.name = "viewer"
        self.settings = plugin_settings or {}
        # 表示モジュールを動的にロード
        self._load_display_module()
    
    def _load_display_module(self):
        """settings.jsonのdisplay_moduleから表示モジュールを動的にロードする"""
        display_module_path = self.settings.get("display_module")
        
        # モジュールの読み込み
        spec = importlib.util.spec_from_file_location("display_module", display_module_path)
        display_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(display_module)
        
        # display_module関数を取得
        self.display_module_func = display_module.display_module
    
    def clear_screen(self):
        """画面クリア（Windows対応）"""
        sys.stdout.write('\033[3J\033[H\033[2J')  # スクロールバック+カーソル移動+画面クリア
        sys.stdout.flush()
        time.sleep(0.1) 
    
    def on_message_update(self, app_data):
        """メッセージ更新時の処理 - 画面をクリアして全メッセージを表示"""
        # 画面をクリア
        self.clear_screen()
        
        # display_module関数を呼び出して表示文字列を取得
        display_text = self.display_module_func(app_data.messages)
        
        # 文字列を表示
        print(display_text)
    
    def cleanup(self):
        """クリーンアップ処理"""
        pass