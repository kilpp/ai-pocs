use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SafetyConfig {
    pub allowed_domains: Vec<String>,
    pub max_depth: usize,
    pub max_requests: usize,
    pub rate_limit_ms: u64,
    pub blocked_content_patterns: Vec<String>,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            allowed_domains: vec![],
            max_depth: 3,
            max_requests: 20,
            rate_limit_ms: 1000,
            blocked_content_patterns: vec![],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    pub safety: SafetyConfig,
    pub user_agent: String,
    pub request_timeout_secs: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            safety: SafetyConfig::default(),
            user_agent: "AutonomousWebAgent-POC/0.1".to_string(),
            request_timeout_secs: 30,
        }
    }
}
