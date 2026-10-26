use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use anyhow::{Context, Result};
use toml_edit::{value, ArrayOfTables, DocumentMut, Item, Table};

use crate::config::Config;

fn wizard_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(".wizard")
}

/// Point wizard's openai-kind provider at the tunnel and store the key the way
/// wizard reads it: `[keys] <provider>` in credentials.toml (0600).
pub fn configure(cfg: &Config, key: &str) -> Result<PathBuf> {
    let dir = wizard_dir();
    let path = dir.join("config.toml");
    let text = fs::read_to_string(&path).unwrap_or_default();
    let mut doc: DocumentMut = text.parse().with_context(|| format!("parsing {}", path.display()))?;

    if !doc.contains_key("providers") {
        doc.insert("providers", Item::ArrayOfTables(ArrayOfTables::new()));
    }
    let providers = doc["providers"]
        .as_array_of_tables_mut()
        .context("`providers` in wizard config is not an array of tables")?;
    let existing = providers
        .iter_mut()
        .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(cfg.wizard_provider.as_str()));
    let table = match existing {
        Some(t) => t,
        None => {
            providers.push(Table::new());
            providers.iter_mut().last().unwrap()
        }
    };
    table["name"] = value(cfg.wizard_provider.as_str());
    table["kind"] = value("openai");
    table["base_url"] = value(cfg.local_url());
    table["model"] = value(cfg.model.as_str());
    table["model_pinned"] = value(true);
    fs::create_dir_all(&dir)?;
    fs::write(&path, doc.to_string())?;

    let cred_path = dir.join("credentials.toml");
    let cred_text = fs::read_to_string(&cred_path).unwrap_or_default();
    let mut creds: DocumentMut = cred_text.parse().with_context(|| format!("parsing {}", cred_path.display()))?;
    if !creds.contains_key("keys") {
        creds.insert("keys", Item::Table(Table::new()));
    }
    creds["keys"][cfg.wizard_provider.as_str()] = value(key);
    fs::write(&cred_path, creds.to_string())?;
    fs::set_permissions(&cred_path, fs::Permissions::from_mode(0o600))?;
    Ok(path)
}
