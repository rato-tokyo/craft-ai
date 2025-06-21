import os
from langchain_core.messages import SystemMessage

class Plugin:
    def __init__(self, app_data, plugin_settings=None):
        self.name = "system_prompt_from_flags"
        self.settings = plugin_settings or {}
    
    def on_flag(self, app_data):
        """フラグ処理メイン"""
        # ユーザー入力後にのみ実行
        if app_data["flags"]["after_user_input"]:
            self._update_system_prompt(app_data)
            # AIの準備ができたことを示すフラグを設定
            app_data["flags"]["ready_for_ai"] = True
    
    def _update_system_prompt(self, app_data):
        """フラグに基づいてシステムプロンプトを更新"""
        prompts = []
        
        # フラグを確認し、対応するプロンプトファイルを読み込む
        for flag_name, flag_value in app_data["flags"].items():
            if flag_value and flag_name in self.settings:
                prompt_path = self.settings[flag_name]
                if os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompts.append(f.read())
        
        # プロンプトが見つかった場合、システムプロンプトを更新
        if prompts:
            # プロンプトを連結
            new_system_prompt = "\n\n".join(prompts)
            
            # システムプロンプトを更新
            app_data.system_prompt = new_system_prompt
            
            # 会話履歴の先頭のSystemMessageを更新
            if app_data.messages and isinstance(app_data.messages[0], SystemMessage):
                app_data.messages[0] = SystemMessage(content=new_system_prompt)
            elif app_data.messages:
                # SystemMessageが先頭にない場合は、先頭に挿入
                app_data.messages.insert(0, SystemMessage(content=new_system_prompt))
            else:
                # メッセージが空の場合は、SystemMessageを追加
                app_data.messages.append(SystemMessage(content=new_system_prompt))
    
    def cleanup(self):
        """クリーンアップ処理"""
        pass 