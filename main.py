import argparse
import uvicorn
from app import app

def main():
    parser = argparse.ArgumentParser(description="AI Training System")
    parser.add_argument("--serve", action="store_true", help="Enable web GUI server")
    parser.add_argument("--gui-port", type=int, default=3000, help="GUI server port")
    parser.add_argument("--ssl-key", help="SSL key file")
    parser.add_argument("--ssl-cert", help="SSL certificate file")
    args = parser.parse_args()

    if args.serve:
        ssl_context = (args.ssl_cert, args.ssl_key) if args.ssl_cert and args.ssl_key else None
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=args.gui_port,
            ssl_keyfile=args.ssl_key,
            ssl_certfile=args.ssl_cert,
            reload=True
        )
    else:
        print("Running in API-only mode. Use --serve to start the web GUI.")

if __name__ == "__main__":
    main()
