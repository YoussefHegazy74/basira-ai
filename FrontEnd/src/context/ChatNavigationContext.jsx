import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";

const ChatNavigationContext = createContext(null);

const EMPTY_HISTORY = {
  conversations: [],
  activeId: null,
  loaded: true,
  busy: false,
};

export function ChatNavigationProvider({ children }) {
  const [history, setHistoryState] = useState(EMPTY_HISTORY);
  const actionsRef = useRef(null);

  const setHistory = useCallback((nextHistory) => {
    setHistoryState(nextHistory);
  }, []);

  const setActions = useCallback((nextActions) => {
    actionsRef.current = nextActions;
  }, []);

  const actions = useMemo(
    () => ({
      newChat: () => actionsRef.current?.newChat?.(),
      open: (item) => actionsRef.current?.open?.(item),
      rename: (item) => actionsRef.current?.rename?.(item),
      remove: (item) => actionsRef.current?.remove?.(item),
    }),
    []
  );

  const value = useMemo(
    () => ({
      ...history,
      setHistory,
      setActions,
      actions,
    }),
    [history, setHistory, setActions, actions]
  );

  return (
    <ChatNavigationContext.Provider value={value}>
      {children}
    </ChatNavigationContext.Provider>
  );
}

export function useChatNavigation() {
  const context = useContext(ChatNavigationContext);
  if (!context) {
    throw new Error("useChatNavigation must be used inside ChatNavigationProvider");
  }
  return context;
}
