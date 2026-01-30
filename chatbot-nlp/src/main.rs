use chatbot_nlp::chatbot::Chatbot;
use std::io::{self, Write};
use std::collections::HashMap;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       NLP Chatbot - Traditional Techniques Demo         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    println!("Features:");
    println!("  ✓ Intent Recognition");
    println!("  ✓ Entity Extraction");
    println!("  ✓ Conversation Management");
    println!("  ✓ Context Tracking\n");
    
    println!("Try saying things like:");
    println!("  - 'Hello!'");
    println!("  - 'Book an appointment for tomorrow at 3pm'");
    println!("  - 'What's the weather in New York?'");
    println!("  - 'I want to order food'");
    println!("  - 'Help'");
    println!("  - 'Show context' (to see conversation history)");
    println!("  - 'Bye' (to exit)\n");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let mut chatbot = Chatbot::new();
    let session_id = "main-session";
    
    loop {
        print!("You: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        // Special commands
        if input.eq_ignore_ascii_case("show context") {
            show_context(&chatbot, session_id);
            continue;
        }
        
        if input.eq_ignore_ascii_case("clear") || input.eq_ignore_ascii_case("reset") {
            chatbot.end_conversation(session_id);
            println!("Bot: Conversation context cleared.\n");
            continue;
        }
        
        // Process the message
        let response = chatbot.process_message(session_id, input);
        println!("Bot: {}\n", response);
        
        // Check if user wants to exit
        if input.to_lowercase().contains("bye") || 
           input.to_lowercase().contains("exit") || 
           input.to_lowercase().contains("quit") {
            println!("Thank you for chatting! Goodbye!\n");
            break;
        }
    }
    
    // Show final conversation summary
    show_conversation_summary(&chatbot, session_id);
}

fn show_context(chatbot: &Chatbot, session_id: &str) {
    if let Some(context) = chatbot.get_conversation_context(session_id) {
        println!("\n═══════════════ Conversation Context ═══════════════");
        println!("Session ID: {}", context.session_id);
        println!("Started: {}", context.started_at.format("%Y-%m-%d %H:%M:%S"));
        
        if let Some(last_intent) = &context.last_intent {
            println!("Last Intent: {}", last_intent);
        }
        
        if !context.context_data.is_empty() {
            println!("\nContext Data:");
            for (key, value) in &context.context_data {
                println!("  • {}: {}", key, value);
            }
        }
        
        println!("\nConversation History ({} turns):", context.conversation_history.len());
        for (i, turn) in context.conversation_history.iter().enumerate() {
            println!("  {}. You: {}", i + 1, turn.user_input);
            println!("     Bot: {}", turn.bot_response);
            println!("     Intent: {} | Entities: {:?}", turn.intent, turn.entities);
            println!();
        }
        println!("═══════════════════════════════════════════════════════\n");
    } else {
        println!("No conversation context found.\n");
    }
}

fn show_conversation_summary(chatbot: &Chatbot, session_id: &str) {
    if let Some(context) = chatbot.get_conversation_context(session_id) {
        println!("\n═══════════════ Conversation Summary ═══════════════");
        println!("Total turns: {}", context.conversation_history.len());
        
        // Count intents
        let mut intent_counts: HashMap<String, usize> = HashMap::new();
        for turn in &context.conversation_history {
            *intent_counts.entry(turn.intent.clone()).or_insert(0) += 1;
        }
        
        println!("\nIntent Distribution:");
        for (intent, count) in intent_counts {
            println!("  • {}: {}", intent, count);
        }
        
        println!("═══════════════════════════════════════════════════════\n");
    }
}

