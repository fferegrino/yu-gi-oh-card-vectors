terraform {
    backend "s3" {
        bucket = "delft-windbag-espy-leak"
        key    = "metaflow.tfstate"
        region = "eu-west-1"
    }

  required_version = ">= 1.0"
}
