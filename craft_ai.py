#!/usr/bin/env python3
"""
Craft AI - LangGraph完全依存型AIチャットアプリケーション

クリーンアーキテクチャに基づいた設計：
- 状態管理: StateManager
- システムプロンプト管理: SystemPromptManager  
- 設定管理: ConfigManager
- AI処理: AIProcessor
"""

import json
import importlib.util
import os
from typing import Dict, List, Any, Optional, Annotated, Protocol
from pathlib import Path
from dotenv import load_dotenv

# 必須ライブラリのimport（fail fast）
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class Plugin(Protocol):
    """プラグインのプロトコル定義"""
    def on_message_update(self, app_data: Any, messages: List[BaseMessage]) -> None: ...
    def cleanup(self) -> None: ...


class Tool(Protocol):
    """ツールのプロトコル定義"""
    functions: List[Any]


class SystemPromptManager:
    """システムプロンプト管理クラス"""
    
    def __init__(self, base_prompt_path: str = "base_prompt.md"):
        self.base_prompt_path = Path(base_prompt_path)
        self._base_prompt: Optional[str] = None
    
    def get_base_prompt(self) -> str:
        """ベースシステムプロンプトを取得（キャッシュ付き）"""
        if self._base_prompt is None:
            self._base_prompt = self._load_base_prompt()
        return self._base_prompt
    
    def _load_base_prompt(self) -> str:
        """ベースプロンプトをファイルから読み込み"""
        if self.base_prompt_path.exists():
            return self.base_prompt_path.read_text(encoding="utf-8")
        else:
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """デフォルトプロンプトを返す"""
        return """# 基本情報
・日本語で回答してください
・開発環境はWindows11で、PowerShell7を使用できます。

# ツールを呼び出し際は、ツール呼び出します！！と宣言してください。"""
    
    def generate_dynamic_prompt(self, latest_message: BaseMessage) -> str:
        """メッセージ内容に基づいて動的にシステムプロンプトを生成"""
        base_prompt = self.get_base_prompt()
        
        if isinstance(latest_message, HumanMessage):
            return self._generate_user_context_prompt(base_prompt, latest_message)
        elif isinstance(latest_message, AIMessage):
            return self._generate_ai_context_prompt(base_prompt, latest_message)
        
        return base_prompt
    
    def _generate_user_context_prompt(self, base_prompt: str, message: HumanMessage) -> str:
        """ユーザーメッセージに基づくコンテキスト生成"""
        user_input = message.content.lower()
        additional_contexts = []
        
        if any(keyword in user_input for keyword in ["mono", "tool", "edit_mono"]):
            additional_contexts.append("追加指示：mono_toolの使用が想定されます。適切にツールを活用してください。")
        
        if "test" in user_input:
            additional_contexts.append("追加指示：テスト実行が要求されています。結果を適切に解釈してください。")
        
        if additional_contexts:
            return base_prompt + "\n\n" + "\n\n".join(additional_contexts)
        return base_prompt
    
    def _generate_ai_context_prompt(self, base_prompt: str, message: AIMessage) -> str:
        """AI応答に基づくコンテキスト生成"""
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_names = [tool_call.get('name', '') for tool_call in message.tool_calls]
            if tool_names:
                additional_prompt = f"追加指示：以下のツールが使用されました: {', '.join(tool_names)}。結果を適切に解釈し、ユーザーに分かりやすく説明してください。"
                return base_prompt + "\n\n" + additional_prompt
        return base_prompt


class StateManager:
    """状態管理クラス"""
    
    def __init__(self, system_prompt_manager: SystemPromptManager):
        self.system_prompt_manager = system_prompt_manager
        self._current_state: List[BaseMessage] = []
        self._plugin_registry: Dict[str, Plugin] = {}
        self._initialize_state()
    
    def _initialize_state(self) -> None:
        """状態を初期化"""
        base_prompt = self.system_prompt_manager.get_base_prompt()
        self._current_state = [SystemMessage(content=base_prompt)]
    
    def register_plugin(self, name: str, plugin: Plugin) -> None:
        """プラグインを登録"""
        self._plugin_registry[name] = plugin
    
    def get_current_messages(self) -> List[BaseMessage]:
        """現在のメッセージ状態を取得"""
        return self._current_state.copy()
    
    def update_system_prompt(self, latest_message: BaseMessage) -> None:
        """システムプロンプトを更新"""
        new_prompt = self.system_prompt_manager.generate_dynamic_prompt(latest_message)
        if self._current_state:
            self._current_state[0] = SystemMessage(content=new_prompt)
        else:
            self._current_state = [SystemMessage(content=new_prompt)]
    
    def update_messages(self, new_messages: List[BaseMessage]) -> None:
        """メッセージ状態を更新"""
        self._current_state = new_messages
    
    def execute_plugins(self, app_data: Any, messages: List[BaseMessage]) -> None:
        """登録されたプラグインを実行"""
        for plugin_name, plugin in self._plugin_registry.items():
            if hasattr(plugin, 'on_message_update'):
                plugin.on_message_update(app_data, messages)





class CustomState(TypedDict):
    """カスタム状態定義"""
    messages: Annotated[List[BaseMessage], add_messages]


class CustomChatAnthropic(BaseChatModel):
    """カスタムChatAnthropicモデル"""
    
    def __init__(self, model: str, max_tokens: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self._validate_api_key()
        self._internal_llm = ChatAnthropic(
            model=model, 
            max_tokens=max_tokens,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
        )
    
    def _validate_api_key(self) -> None:
        """APIキーの存在確認"""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("環境変数ANTHROPIC_API_KEYが設定されていません")
    
    @property
    def _llm_type(self) -> str:
        return "custom_chat_anthropic"
    
    def bind_tools(self, tools: List[Any]) -> "CustomChatAnthropic":
        """ツールをバインド"""
        if tools:
            self._internal_llm = self._internal_llm.bind_tools(tools)
        return self
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        return self._internal_llm.invoke(messages, **kwargs)
    
    def stream(self, messages: List[BaseMessage], **kwargs):
        return self._internal_llm.stream(messages, **kwargs)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._internal_llm._generate(messages, stop, run_manager, **kwargs)


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_file}")
        
        with open(self.config_file, encoding="utf-8") as f:
            return json.load(f)
    
    def get_model_name(self) -> str:
        """モデル名を取得"""
        if "model" not in self.settings:
            raise ValueError("settings.jsonにmodelが設定されていません")
        return self.settings["model"]
    
    def get_plugin_configs(self) -> List[Dict[str, Any]]:
        """プラグイン設定を取得"""
        if "plugins" not in self.settings:
            raise ValueError("settings.jsonにpluginsが設定されていません")
        return self.settings["plugins"]
    
    def get_tool_configs(self) -> List[Dict[str, Any]]:
        """ツール設定を取得"""
        if "tools" not in self.settings:
            raise ValueError("settings.jsonにtoolsが設定されていません")
        return self.settings["tools"]
    
    def get_inputs(self) -> List[str]:
        """入力配列を取得"""
        if "inputs" not in self.settings:
            raise ValueError("settings.jsonにinputsが設定されていません")
        return self.settings["inputs"]


class ComponentLoader:
    """コンポーネント読み込みクラス"""
    
    @staticmethod
    def load_plugins(plugin_configs: List[Dict[str, Any]], app_data: Any) -> Dict[str, Plugin]:
        """プラグインを読み込み"""
        plugins = {}
        
        for plugin_config in plugin_configs:
            if "path" not in plugin_config:
                raise ValueError("プラグイン設定にpathが設定されていません")
            
            plugin_path = plugin_config["path"]
            plugin_settings = plugin_config.get("settings", {})
            plugin_name = os.path.basename(os.path.dirname(plugin_path))
            
            # 削除されたプラグインはスキップ
            if plugin_name == "system_prompt_from_flags":
                continue
            
            try:
                plugin = ComponentLoader._load_module_component(
                    plugin_path, "Plugin", app_data, plugin_settings
                )
                plugins[plugin_name] = plugin
            except Exception as e:
                print(f"警告: プラグイン '{plugin_name}' の読み込みに失敗: {e}")
        
        return plugins
    
    @staticmethod
    def load_tools(tool_configs: List[Dict[str, Any]], app_data: Any) -> Dict[str, Tool]:
        """ツールを読み込み"""
        tools = {}
        
        for tool_config in tool_configs:
            if "path" not in tool_config:
                raise ValueError("ツール設定にpathが設定されていません")
            
            tool_path = tool_config["path"]
            tool_settings = tool_config.get("settings", {})
            tool_name = os.path.basename(os.path.dirname(tool_path))
            
            try:
                tool = ComponentLoader._load_module_component(
                    tool_path, "Tool", {"settings": tool_settings, "app_data": app_data}
                )
                
                # ツール関数を抽出
                tool.functions = ComponentLoader._extract_tool_functions(tool_path)
                tools[tool_name] = tool
            except Exception as e:
                print(f"警告: ツール '{tool_name}' の読み込みに失敗: {e}")
        
        return tools
    
    @staticmethod
    def _load_module_component(module_path: str, class_name: str, *args, **kwargs):
        """モジュールからコンポーネントを読み込み"""
        spec = importlib.util.spec_from_file_location("component", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        component_class = getattr(module, class_name)
        return component_class(*args, **kwargs)
    
    @staticmethod
    def _extract_tool_functions(tool_path: str) -> List[Any]:
        """ツールモジュールから関数を抽出"""
        spec = importlib.util.spec_from_file_location("tool", tool_path)
        tool_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool_module)
        
        functions = []
        for name in dir(tool_module):
            obj = getattr(tool_module, name)
            if hasattr(obj, '__call__') and hasattr(obj, 'name'):
                functions.append(obj)
        return functions


class AIProcessor:
    """AI処理クラス"""
    
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager):
        self.config_manager = config_manager
        self.state_manager = state_manager
        self._agent = None
        self._initialized = False
    
    def initialize(self, app_data: Any) -> None:
        """AI処理を初期化"""
        if self._initialized:
            return
        
        # AppDataの参照を保存
        self._app_data = app_data
        
        # LLMを初期化
        llm = CustomChatAnthropic(
            model=self.config_manager.get_model_name(),
            max_tokens=4000
        )
        
        # ツールを読み込んでバインド
        tools = ComponentLoader.load_tools(self.config_manager.get_tool_configs(), app_data)
        valid_tools = self._extract_valid_tools(tools)
        
        if valid_tools:
            llm = llm.bind_tools(valid_tools)
        
        # LangGraphワークフローを構築
        self._agent = self._build_workflow(llm, valid_tools)
        self._initialized = True
    
    def _extract_valid_tools(self, tools: Dict[str, Tool]) -> List[Any]:
        """有効なツール関数を抽出"""
        valid_tools = []
        for tool_name, tool in tools.items():
            if hasattr(tool, 'functions'):
                for func in tool.functions:
                    if hasattr(func, '__call__') and hasattr(func, 'name'):
                        valid_tools.append(func)
        return valid_tools
    
    def _build_workflow(self, llm: CustomChatAnthropic, tools: List[Any]):
        """LangGraphワークフローを構築"""
        def llm_node(state: CustomState):
            messages = state["messages"]
            response = llm.invoke(messages)
            # プラグインを実行
            self.state_manager.execute_plugins(self._app_data, state["messages"] + [response])
            return {"messages": [response]}
        
        workflow = StateGraph(CustomState)
        workflow.add_node("llm", llm_node)
        
        if tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges("llm", tools_condition)
            workflow.add_edge("tools", "llm")
        
        workflow.add_edge(START, "llm")
        workflow.add_edge("llm", END)
        
        return workflow.compile()
    
    def process_user_input(self, user_input: str, app_data: Any) -> None:
        """ユーザー入力を処理"""
        if not self._initialized or self._agent is None:
            raise RuntimeError("AIが初期化されていません")
        
        print(f"[DEBUG] ユーザー入力処理開始: {user_input}")
        
        # ユーザーメッセージを作成
        user_message = HumanMessage(content=user_input)
        
        # システムプロンプトを動的更新
        self.state_manager.update_system_prompt(user_message)
        
        # 現在のメッセージ状態を取得
        current_messages = self.state_manager.get_current_messages()
        
        # AI応答を生成（LangGraphが自動でプラグインを実行）
        result = self._agent.invoke({
            "messages": current_messages + [user_message]
        })
        
        print(f"[DEBUG] AI応答生成完了 - 最終メッセージ数: {len(result['messages'])}")
        
        # 状態を更新
        self.state_manager.update_messages(result["messages"])
        
        # AI応答後のシステムプロンプト更新
        if result["messages"] and isinstance(result["messages"][-1], AIMessage):
            self.state_manager.update_system_prompt(result["messages"][-1])


class AppData:
    """アプリケーションデータ管理（シングルトンパターン）"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_file: str = None):
        if cls._instance is None:
            cls._instance = super(AppData, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: str = None):
        if self._initialized:
            return
        
        if config_file is None:
            raise ValueError("初回初期化時にはconfig_fileが必要です")
        
        # 環境変数を読み込み
        load_dotenv()
        
        # コンポーネントを初期化
        self.config_manager = ConfigManager(config_file)
        self.system_prompt_manager = SystemPromptManager()
        self.state_manager = StateManager(self.system_prompt_manager)
        self.ai_processor = AIProcessor(self.config_manager, self.state_manager)
        
        # プラグインを読み込み
        self.plugins = ComponentLoader.load_plugins(
            self.config_manager.get_plugin_configs(), self
        )
        
        # プラグインを状態管理に登録
        for name, plugin in self.plugins.items():
            self.state_manager.register_plugin(name, plugin)
        
        # ツールを読み込み
        self.tools = ComponentLoader.load_tools(
            self.config_manager.get_tool_configs(), self
        )
        
        # 入力配列を設定
        self.inputs = self.config_manager.get_inputs()
        
        # AI処理を初期化
        self.ai_processor.initialize(self)
        
        self._initialized = True
    
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)
    
    @property
    def messages(self) -> List[BaseMessage]:
        """現在のメッセージ状態を取得"""
        return self.state_manager.get_current_messages()
    
    @property
    def model(self) -> str:
        """モデル名を取得（後方互換性）"""
        return self.config_manager.get_model_name()


def cleanup_plugins(app_data: AppData) -> None:
    """プラグインのクリーンアップ"""
    for plugin_name, plugin in app_data.plugins.items():
        if hasattr(plugin, 'cleanup'):
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"警告: プラグイン '{plugin_name}' のクリーンアップに失敗: {e}")


def main() -> None:
    """メイン関数"""
    try:
        app_data = AppData("settings.json")
        
        input_index = 0
        while input_index < len(app_data.inputs):
            user_input = app_data.inputs[input_index]
            input_index += 1
            
            if user_input.lower() == "exit":
                break
            
            # ユーザー入力を処理
            app_data.ai_processor.process_user_input(user_input, app_data)
    
    except Exception as e:
        print(f"エラー: {e}")
        raise
    finally:
        if 'app_data' in locals():
            cleanup_plugins(app_data)


if __name__ == "__main__":
    main() 