import queue

# Shared, process-wide thread-safe queue for incoming TCP messages.
# Import this from other modules so there is a single queue instance.
GLOBAL_INCOMING_QUEUE = queue.Queue()
