use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::intent::Intent;
use crate::entity::Entity;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub user_input: String,
    pub bot_response: String,
    pub intent: String,
    pub entities: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    pub session_id: String,
    pub user_name: Option<String>,
    pub conversation_history: VecDeque<ConversationTurn>,
    pub context_data: HashMap<String, String>,
    pub started_at: DateTime<Utc>,
    pub last_intent: Option<String>,
}

impl ConversationContext {
    pub fn new(session_id: String) -> Self {
        ConversationContext {
            session_id,
            user_name: None,
            conversation_history: VecDeque::with_capacity(10),
            context_data: HashMap::new(),
            started_at: Utc::now(),
            last_intent: None,
        }
    }
    
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.last_intent = Some(turn.intent.clone());
        
        // Keep only last 10 turns
        if self.conversation_history.len() >= 10 {
            self.conversation_history.pop_front();
        }
        
        self.conversation_history.push_back(turn);
    }
    
    pub fn set_context(&mut self, key: String, value: String) {
        self.context_data.insert(key, value);
    }
    
    pub fn get_context(&self, key: &str) -> Option<&String> {
        self.context_data.get(key)
    }
    
    pub fn get_last_turns(&self, count: usize) -> Vec<&ConversationTurn> {
        self.conversation_history
            .iter()
            .rev()
            .take(count)
            .rev()
            .collect()
    }
}

pub struct ConversationManager {
    sessions: HashMap<String, ConversationContext>,
}

impl ConversationManager {
    pub fn new() -> Self {
        ConversationManager {
            sessions: HashMap::new(),
        }
    }
    
    pub fn get_or_create_session(&mut self, session_id: String) -> &mut ConversationContext {
        self.sessions
            .entry(session_id.clone())
            .or_insert_with(|| ConversationContext::new(session_id))
    }
    
    pub fn get_session(&self, session_id: &str) -> Option<&ConversationContext> {
        self.sessions.get(session_id)
    }
    
    pub fn get_session_mut(&mut self, session_id: &str) -> Option<&mut ConversationContext> {
        self.sessions.get_mut(session_id)
    }
    
    pub fn end_session(&mut self, session_id: &str) -> Option<ConversationContext> {
        self.sessions.remove(session_id)
    }
    
    pub fn record_turn(
        &mut self,
        session_id: &str,
        user_input: String,
        bot_response: String,
        intent: Intent,
        entities: Vec<Entity>,
    ) {
        if let Some(context) = self.get_session_mut(session_id) {
            let turn = ConversationTurn {
                user_input,
                bot_response,
                intent: format!("{:?}", intent),
                entities: entities.iter().map(|e| e.value.clone()).collect(),
                timestamp: Utc::now(),
            };
            context.add_turn(turn);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conversation_context() {
        let mut context = ConversationContext::new("test-session".to_string());
        
        let turn = ConversationTurn {
            user_input: "Hello".to_string(),
            bot_response: "Hi there!".to_string(),
            intent: "Greeting".to_string(),
            entities: vec![],
            timestamp: Utc::now(),
        };
        
        context.add_turn(turn);
        assert_eq!(context.conversation_history.len(), 1);
    }
    
    #[test]
    fn test_context_storage() {
        let mut context = ConversationContext::new("test-session".to_string());
        context.set_context("user_preference".to_string(), "coffee".to_string());
        
        assert_eq!(
            context.get_context("user_preference"),
            Some(&"coffee".to_string())
        );
    }
}
