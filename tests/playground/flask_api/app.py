"""Simple Flask API for testing directory traversal understanding."""

from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory store
_items = {}
_next_id = 1


@app.route("/api/items", methods=["GET"])
def list_items():
    """List all items with optional filtering."""
    category = request.args.get("category")
    items = list(_items.values())
    if category:
        items = [i for i in items if i.get("category") == category]
    return jsonify({"items": items, "total": len(items)})


@app.route("/api/items", methods=["POST"])
def create_item():
    """Create a new item."""
    global _next_id
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "name is required"}), 400
    
    item = {
        "id": _next_id,
        "name": data["name"],
        "category": data.get("category", "general"),
        "description": data.get("description", ""),
    }
    _items[_next_id] = item
    _next_id += 1
    return jsonify(item), 201


@app.route("/api/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    """Get a single item by ID."""
    item = _items.get(item_id)
    if not item:
        return jsonify({"error": "not found"}), 404
    return jsonify(item)


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    """Delete an item by ID."""
    if item_id not in _items:
        return jsonify({"error": "not found"}), 404
    del _items[item_id]
    return jsonify({"deleted": True})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "items_count": len(_items)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
