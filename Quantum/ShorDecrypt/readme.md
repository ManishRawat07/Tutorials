Usage:

Start the sniffing tool: sudo python sniff.py

Start the server: python server.py

Start the client: python client.py

Send some fictitious credentials.

The sniffing tool should intercept the public key, factor it with Shor's algorithm, and decrypt the emails and credentials.
Options:

--debug enables debug-level logging.

--aer (for sniff.py) uses the Aer backend. If this flag is not given, the default is the IBMQ QASM simulator.

--cache (for sniff.py) uses cached jobs.
