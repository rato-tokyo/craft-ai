#!/usr/bin/env python3
"""
Craft AI - カスタムreducer関数によるメッセージフック型AIチャットアプリケーション

メッセージが追加されるたびにプラグインが自動実行される仕様
"""

import json
import importlib.util
import os
import sys
from typing import Dict, List, Any, Optional, Annotated
from dotenv import load_dotenv

# 必須ライブラリのimport（fail fast）
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


# グローバル変数としてAppDataのインスタンスを保持
_global_app_data = None


def custom_message_reducer(existing: List[BaseMessage], updates: List[BaseMessage]) -> List[BaseMessage]:
    """カスタムメッセージreducer関数 - メッセージ追加時にプラグインを自動実行"""
    global _global_app_data
    
    # 標準のadd_messagesを使用してメッセージを追加
    result = add_messages(existing, updates)
    
    # 新しいメッセージが追加された場合のみプラグインを実行
    if len(result) > len(existing) and _global_app_data is not None:
        # プラグインを実行（画面クリア + 全メッセージ表示）
        _execute_plugins_on_message_update(_global_app_data)
    
    return result


def _execute_plugins_on_message_update(app_data) -> None:
    """メッセージ更新時にプラグインを実行"""
    for plugin_name, plugin in app_data.plugins.items():
        if hasattr(plugin, 'on_message_update'):
            plugin.on_message_update(app_data)


class CustomMessagesState(TypedDict):
    """カスタムメッセージ状態 - メッセージ追加時にフックを実行"""
    messages: Annotated[List[BaseMessage], custom_message_reducer]


class CustomChatAnthropic(BaseChatModel):
    """高速化対応のカスタムChatAnthropicモデル"""
    
    def __init__(self, model: str, max_tokens: int = 4000, **kwargs):
        super().__init__(**kwargs)
        
        # 環境変数からAPIキーを取得（fail fast）
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("環境変数ANTHROPIC_API_KEYが設定されていません")
        
        # 内部でChatAnthropicを使用
        self._internal_llm = ChatAnthropic(
            model=model, 
            max_tokens=max_tokens,
            anthropic_api_key=api_key
        )
    
    @property
    def _llm_type(self) -> str:
        """モデルタイプを返す"""
        return "custom_chat_anthropic"
    
    def bind_tools(self, tools: List) -> "CustomChatAnthropic":
        """ツールをバインドする"""
        if tools:
            self._internal_llm = self._internal_llm.bind_tools(tools)
        return self
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """メッセージを送信してAI応答を取得"""
        return self._internal_llm.invoke(messages, **kwargs)
    
    def stream(self, messages: List[BaseMessage], **kwargs):
        """ストリーミングでメッセージを送信"""
        return self._internal_llm.stream(messages, **kwargs)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同期的にチャット応答を生成"""
        result = self._internal_llm._generate(messages, stop, run_manager, **kwargs)
        return result


class AppData:
    """アプリケーションデータ管理（辞書形式アクセス）- シングルトンパターン"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_file: str = None):
        """シングルトンインスタンスを返す"""
        if cls._instance is None:
            cls._instance = super(AppData, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: str = None):
        """JSONファイルから初期化（初回のみ）"""
        if self._initialized:
            return
        
        if config_file is None:
            raise ValueError("初回初期化時にはconfig_fileが必要です")
        
        # .envファイルの読み込み
        load_dotenv()
        
        with open(config_file, encoding="utf-8") as f:
            settings = json.load(f)
        
        # 設定値を使用して各コンポーネントを初期化
        self._load_model(settings)
        self._load_messages()
        self._load_plugins(settings)
        self._load_tools(settings)
        self._load_inputs(settings)
        
        # グローバル変数にインスタンスを設定
        global _global_app_data
        _global_app_data = self
        
        # AI初期化（fail fast）
        AI.initialize_from_settings(self)
        
        # 初期化完了フラグ
        self._initialized = True
    
    def __getitem__(self, key: str) -> Any:
        """辞書形式でのアクセスを可能にする"""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """辞書形式での設定を可能にする"""
        setattr(self, key, value)
    
    def _load_model(self, settings: Dict[str, Any]) -> None:
        """モデル設定読み込み（fail fast）"""
        if "model" not in settings:
            raise ValueError("settings.jsonにmodelが設定されていません")
        self.model = settings["model"]
    
    def _load_plugins(self, settings: Dict[str, Any]) -> None:
        """プラグイン読み込み（fail fast）"""
        if "plugins" not in settings:
            raise ValueError("settings.jsonにpluginsが設定されていません")
        
        self.plugins = {}
        
        for plugin_config in settings["plugins"]:
            if "path" not in plugin_config:
                raise ValueError("プラグイン設定にpathが設定されていません")
            
            plugin_path = plugin_config["path"]
            plugin_settings = plugin_config.get("settings", {})
            
            # プラグイン名を取得（パスからディレクトリ名を抽出）
            plugin_name = os.path.basename(os.path.dirname(plugin_path))
            
            # モジュールの読み込み
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # プラグインインスタンス作成
            plugin = plugin_module.Plugin(self, plugin_settings)
            
            # プラグインをディクショナリに追加
            self.plugins[plugin_name] = plugin
    
    def _load_tools(self, settings: Dict[str, Any]) -> None:
        """ツール読み込み（fail fast）"""
        if "tools" not in settings:
            raise ValueError("settings.jsonにtoolsが設定されていません")
        
        self.tools = {}
        
        for tool_config in settings["tools"]:
            if "path" not in tool_config:
                raise ValueError("ツール設定にpathが設定されていません")
            
            tool_path = tool_config["path"]
            tool_settings = tool_config.get("settings", {})
            
            # ツール名を取得（パスからディレクトリ名を抽出）
            tool_name = os.path.basename(os.path.dirname(tool_path))
            
            # モジュールの読み込み
            spec = importlib.util.spec_from_file_location("tool", tool_path)
            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)
            
            # ツールインスタンス作成
            tool = tool_module.Tool(settings=tool_settings, app_data=self)
            
            # ツール関数をインスタンスに関連付け
            tool.functions = []
            for name in dir(tool_module):
                obj = getattr(tool_module, name)
                if hasattr(obj, '__call__') and hasattr(obj, 'name'):
                    tool.functions.append(obj)
            
            # ツールをディクショナリに追加
            self.tools[tool_name] = tool
    
    def _load_inputs(self, settings: Dict[str, Any]) -> None:
        """入力配列初期化（fail fast）"""
        if "inputs" not in settings:
            raise ValueError("settings.jsonにinputsが設定されていません")
        self.inputs = settings["inputs"]
    
    def _load_messages(self) -> None:
        """会話履歴初期化"""
        # プラグインがmessages[0]のSystemMessageを期待するため、空のSystemMessageで初期化
        self.messages = [SystemMessage(content="")]
    
    def add_message(self, message: BaseMessage) -> None:
        """メッセージを追加 - カスタムreducerを通して処理"""
        if isinstance(message, SystemMessage):
            # SystemMessageの場合は常にmessages[0]を更新
            self.messages[0] = message
        else:
            # その他のメッセージはカスタムreducerを通して追加
            # カスタムreducerが呼ばれるように、新しいメッセージのリストを作成
            updated_messages = custom_message_reducer(self.messages, [message])
            self.messages = updated_messages


class AI:
    """AIグローバル変数クラス"""
    
    _agent = None
    _initialized = False
    _app_data = None
    
    @classmethod
    def _initialize(cls, app_data: AppData) -> None:
        """AIエージェントの初期化"""
        if cls._initialized:
            return
        
        # AppDataの参照を保存
        cls._app_data = app_data
        
        # CustomChatAnthropicの初期化（fail fast）
        llm = CustomChatAnthropic(model=app_data.model, max_tokens=4000)
        
        # ツールのバインド
        valid_tools = []
        for tool_name, tool in app_data.tools.items():
            if hasattr(tool, 'functions'):
                for func in tool.functions:
                    if hasattr(func, '__call__') and hasattr(func, 'name'):
                        valid_tools.append(func)
        
        if valid_tools:
            llm = llm.bind_tools(valid_tools)
        
        # LLMノードの定義
        def llm_node(state: CustomMessagesState):
            """LLMノード: メッセージを受け取り、AIの応答を返す"""
            messages = state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}
        
        # カスタム状態を使用してグラフを構築
        workflow = StateGraph(CustomMessagesState)
        workflow.add_node("llm", llm_node)
        
        if valid_tools:
            workflow.add_node("tools", ToolNode(valid_tools))
            workflow.add_conditional_edges("llm", tools_condition)
            workflow.add_edge("tools", "llm")
        
        workflow.add_edge(START, "llm")
        workflow.add_edge("llm", END)
        
        # エージェントのコンパイル
        cls._agent = workflow.compile()
        cls._initialized = True
    
    @classmethod
    def send(cls) -> str:
        """メッセージを送信してAI応答を取得"""
        if not cls._initialized or cls._agent is None:
            raise RuntimeError("AIが初期化されていません")
        
        # カスタムreducer関数により、メッセージが追加されるたびに自動的にプラグインが実行される
        result = cls._agent.invoke({"messages": cls._app_data.messages})
        
        # 最終的にmessagesを更新（カスタムreducerが自動的にプラグインを実行）
        cls._app_data.messages = result["messages"]
        
        return "AI応答完了"
    
    @classmethod
    def initialize_from_settings(cls, app_data: AppData) -> None:
        """設定からAIを初期化"""
        cls._initialize(app_data)


def get_user_input() -> str:
    """ユーザー入力を取得"""
    return input("ユーザー: ")


def close_plugins(app_data: AppData) -> None:
    """プラグインのクリーンアップ"""
    for plugin_name, plugin in app_data.plugins.items():
        if hasattr(plugin, 'cleanup'):
            plugin.cleanup()


def main() -> None:
    """メイン関数"""
    app_data = AppData("settings.json")
    
    try:
        input_index = 0
        
        while input_index < len(app_data.inputs):
            user_input = app_data.inputs[input_index]
            input_index += 1
            
            if user_input.lower() == "exit":
                break
            
            # ユーザーメッセージを追加（カスタムreducerが自動的にプラグインを実行）
            app_data.add_message(HumanMessage(content=user_input))
            
            # AI応答（カスタムreducerにより各メッセージ追加時に自動的にプラグインが実行される）
            AI.send()
    
    finally:
        close_plugins(app_data)


if __name__ == "__main__":
    main() 