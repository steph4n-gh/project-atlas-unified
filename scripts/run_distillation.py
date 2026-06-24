#!/usr/bin/env python
import sys
# mx.set_default_device(mx.cpu)
from ultrametric_ce.cli.distill import main
if __name__ == "__main__":
    sys.exit(main())
