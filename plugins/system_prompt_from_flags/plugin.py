import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class Plugin:
    def __init__(self, app_data, plugin_settings=None):
        self.name = "system_prompt_from_flags"
        self.settings = plugin_settings or {}
        # 初期システムプロンプトを設定
        self._initialize_system_prompt(app_data)
    
    def _initialize_system_prompt(self, app_data):
        """初期システムプロンプトを設定"""
        if "base" in self.settings:
            prompt_path = self.settings["base"]
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    base_prompt = f.read()
                app_data.messages[0] = SystemMessage(content=base_prompt)
    
    def on_message_update(self, app_data):
        """メッセージ更新時にシステムプロンプトを動的に更新"""
        # 最新のメッセージを取得
        if len(app_data.messages) > 1:
            latest_message = app_data.messages[-1]
            
            # ユーザーメッセージの場合、内容に基づいてシステムプロンプトを更新
            if isinstance(latest_message, HumanMessage):
                self._update_system_prompt_based_on_user_input(app_data, latest_message.content)
            
            # AIメッセージの場合、ツール使用状況に基づいてシステムプロンプトを更新
            elif isinstance(latest_message, AIMessage):
                self._update_system_prompt_based_on_ai_response(app_data, latest_message)
    
    def _update_system_prompt_based_on_user_input(self, app_data, user_input):
        """ユーザー入力に基づいてシステムプロンプトを更新"""
        prompts = []
        
        # ベースプロンプトを常に含める
        if "base" in self.settings:
            prompt_path = self.settings["base"]
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompts.append(f.read())
        
        # ユーザー入力に特定のキーワードが含まれている場合、追加プロンプトを含める
        # 例：mono_toolに関する入力の場合
        if "mono" in user_input.lower() or "tool" in user_input.lower():
            # mono_toolフラグを有効にする
            additional_prompt = "\n\n追加指示：mono_toolの使用が想定されます。適切にツールを活用してください。"
            prompts.append(additional_prompt)
        
        # プロンプトを連結してシステムメッセージを更新
        if prompts:
            new_system_prompt = "\n\n".join(prompts)
            app_data.messages[0] = SystemMessage(content=new_system_prompt)
    
    def _update_system_prompt_based_on_ai_response(self, app_data, ai_message):
        """AI応答に基づいてシステムプロンプトを更新"""
        # ツール呼び出しがある場合の処理
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            # ツール使用後の追加指示をシステムプロンプトに含める
            prompts = []
            
            # ベースプロンプトを含める
            if "base" in self.settings:
                prompt_path = self.settings["base"]
                if os.path.exists(prompt_path):
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompts.append(f.read())
            
            # ツール使用後の追加指示
            tool_names = [tool_call.get('name', '') for tool_call in ai_message.tool_calls]
            if tool_names:
                additional_prompt = f"\n\n追加指示：以下のツールが使用されました: {', '.join(tool_names)}。結果を適切に解釈し、ユーザーに分かりやすく説明してください。"
                prompts.append(additional_prompt)
            
            # プロンプトを連結してシステムメッセージを更新
            new_system_prompt = "\n\n".join(prompts)
            app_data.messages[0] = SystemMessage(content=new_system_prompt)
    
    def cleanup(self):
        """クリーンアップ処理"""
        pass 