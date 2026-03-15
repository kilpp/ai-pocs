use crate::config::AgentConfig;
use crate::error::Result;
use reqwest::Client;
use scraper::{Html, Selector};
use std::time::Duration;
use url::Url;

pub struct Browser {
    client: Client,
}

#[derive(Debug)]
pub struct PageContent {
    pub url: Url,
    pub status: u16,
    pub body: String,
}

#[derive(Debug)]
pub struct ParsedPage {
    pub url: Url,
    pub params: String,
    pub title: Option<String>,
    pub links: Vec<Url>,
    pub text_content: String,
    pub document: Html,
}

impl Browser {
    pub fn new(config: &AgentConfig) -> Result<Self> {
        let client = Client::builder()
            .user_agent(&config.user_agent)
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()?;
        Ok(Self { client })
    }

    pub async fn fetch(&self, url: &Url) -> Result<PageContent> {
        let response = self.client.get(url.as_str()).send().await?;
        let final_url = response.url().clone();
        let status = response.status().as_u16();
        let body = response.text().await?;

        Ok(PageContent {
            url: final_url,
            status,
            body,
        })
    }

    pub fn parse(page: &PageContent) -> ParsedPage {
        let document = Html::parse_document(&page.body);

        let title = Selector::parse("title")
            .ok()
            .and_then(|sel| document.select(&sel).next())
            .map(|el| el.inner_html().trim().to_string());

        let links = Self::extract_links(&document, &page.url);

        let text_content = document.root_element().text().collect::<Vec<_>>().join(" ");

        ParsedPage {
            url: page.url.clone(),
            title,
            links,
            text_content,
            document,
        }
    }

    fn extract_links(document: &Html, base_url: &Url) -> Vec<Url> {
        let selector = match Selector::parse("a[href]") {
            Ok(s) => s,
            Err(_) => return vec![],
        };

        document
            .select(&selector)
            .filter_map(|el| {
                let href = el.value().attr("href")?;
                base_url.join(href).ok()
            })
            .filter(|url| matches!(url.scheme(), "http" | "https"))
            .collect()
    }
}
