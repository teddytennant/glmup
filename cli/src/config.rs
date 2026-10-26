use std::path::PathBuf;

use anyhow::{Context, Result};
use toml_edit::DocumentMut;

/// `~/.config/glmup/config.toml`. Every key is optional; env vars win over the file.
pub struct Config {
    pub host: String,
    pub root: String,
    pub account: String,
    pub partition: String,
    pub local_port: u16,
    pub model: String,
    pub wizard_provider: String,
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = Self::path();
        let doc: DocumentMut = match std::fs::read_to_string(&path) {
            Ok(s) => s.parse().with_context(|| format!("parsing {}", path.display()))?,
            Err(_) => DocumentMut::new(),
        };
        let get = |key: &str, env: &str, default: &str| -> String {
            std::env::var(env)
                .ok()
                .or_else(|| doc.get(key).and_then(|v| v.as_str()).map(str::to_string))
                .unwrap_or_else(|| default.to_string())
        };
        let local_port = std::env::var("GLMUP_LOCAL_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| doc.get("local_port").and_then(|v| v.as_integer()).map(|v| v as u16))
            .unwrap_or(30000);
        Ok(Self {
            host: get("host", "GLMUP_HOST", "ncshare"),
            root: get("root", "GLMUP_ROOT", "/work/ttennant1/glmup"),
            account: get("account", "GLMUP_ACCOUNT", "ncssm"),
            partition: get("partition", "GLMUP_PARTITION", "gpu"),
            local_port,
            model: get("model", "GLMUP_MODEL_NAME", "glm-5.3-flash-uncensored"),
            wizard_provider: get("wizard_provider", "GLMUP_WIZARD_PROVIDER", "sgapi"),
        })
    }

    pub fn path() -> PathBuf {
        dirs::config_dir().unwrap_or_default().join("glmup/config.toml")
    }

    pub fn state_dir() -> PathBuf {
        dirs::state_dir()
            .or_else(dirs::data_local_dir)
            .unwrap_or_else(std::env::temp_dir)
            .join("glmup")
    }

    pub fn local_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.local_port)
    }
}
