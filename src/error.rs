use std::fmt;

#[derive(Debug)]
pub enum MatrixError {
    InvalidInput(&'static str),
    Allocation(String),
    Runtime(String),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => fmt.write_str(message),
            Self::Allocation(message) | Self::Runtime(message) => fmt.write_str(message),
        }
    }
}

impl std::error::Error for MatrixError {}
