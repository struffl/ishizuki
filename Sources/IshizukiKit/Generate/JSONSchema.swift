// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation

/// Expands a JSON Schema into the exact set of compact documents it admits.
///
/// Constrained decoding needs to know, at every byte, which continuations stay inside the
/// schema. For the schemas agent harnesses actually send — an action object over enumerated
/// verbs, a small integer range, a handful of optional keys — that language is finite and
/// small, so enumerating it is both exact and cheap. Schemas whose language is unbounded
/// (a free-form string, an unbounded integer) are reported as unsupported rather than
/// silently approximated.
public enum JSONSchema {
  public enum Unsupported: Error, CustomStringConvertible {
    case unboundedValue(String)
    case tooManyDocuments(Int)
    case malformed(String)

    public var description: String {
      switch self {
      case .unboundedValue(let what):
        return
          "\(what) admits unboundedly many values; constrained decoding needs an enum, "
          + "a bounded integer range, or a boolean"
      case .tooManyDocuments(let limit):
        return "the schema admits more than \(limit) documents"
      case .malformed(let what):
        return "schema is malformed: \(what)"
      }
    }
  }

  public static let documentLimit = 20_000
  public static let integerRangeLimit = 256

  /// Every compact document the schema admits, in a stable order.
  public static func documents(_ schema: [String: Any], limit: Int = documentLimit) throws
    -> [String]
  {
    let values = try expand(schema, limit: limit)
    if values.count > limit { throw Unsupported.tooManyDocuments(limit) }
    return values
  }

  private static func expand(_ schema: [String: Any], limit: Int) throws -> [String] {
    if let constant = schema["const"] { return [render(constant)] }
    if let choices = schema["enum"] as? [Any] { return choices.map(render) }

    switch schema["type"] as? String {
    case "object":
      return try expandObject(schema, limit: limit)
    case "array":
      return try expandArray(schema, limit: limit)
    case "boolean":
      return ["true", "false"]
    case "null":
      return ["null"]
    case "integer":
      return try expandInteger(schema)
    case "string":
      throw Unsupported.unboundedValue("a string without an enum")
    case "number":
      throw Unsupported.unboundedValue("a number")
    case let other:
      throw Unsupported.malformed("unsupported type \(other ?? "none")")
    }
  }

  private static func expandInteger(_ schema: [String: Any]) throws -> [String] {
    guard let minimum = (schema["minimum"] as? NSNumber)?.intValue,
      let maximum = (schema["maximum"] as? NSNumber)?.intValue, maximum >= minimum
    else {
      throw Unsupported.unboundedValue("an integer without an enum or a minimum and maximum")
    }
    let span = maximum - minimum + 1
    guard span <= integerRangeLimit else {
      throw Unsupported.unboundedValue("an integer spanning \(span) values")
    }
    return (minimum...maximum).map(String.init)
  }

  private static func expandArray(_ schema: [String: Any], limit: Int) throws -> [String] {
    guard let items = schema["items"] as? [String: Any] else {
      throw Unsupported.malformed("array without items")
    }
    let lower = (schema["minItems"] as? NSNumber)?.intValue ?? 0
    guard let upper = (schema["maxItems"] as? NSNumber)?.intValue, upper >= lower else {
      throw Unsupported.unboundedValue("an array without maxItems")
    }
    let element = try expand(items, limit: limit)

    var documents: [String] = []
    for count in lower...upper {
      var rows: [String] = count == 0 ? [""] : [""]
      for index in 0..<count {
        var next: [String] = []
        for row in rows {
          for value in element {
            next.append(index == 0 ? value : row + "," + value)
          }
          if next.count > limit { throw Unsupported.tooManyDocuments(limit) }
        }
        rows = next
      }
      documents.append(contentsOf: rows.map { "[" + $0 + "]" })
      if documents.count > limit { throw Unsupported.tooManyDocuments(limit) }
    }
    return documents
  }

  private static func expandObject(_ schema: [String: Any], limit: Int) throws -> [String] {
    guard let properties = schema["properties"] as? [String: Any] else {
      throw Unsupported.malformed("object without properties")
    }
    if (schema["additionalProperties"] as? Bool) == true {
      throw Unsupported.unboundedValue("an object allowing additional properties")
    }
    let required = Set(schema["required"] as? [String] ?? [])

    // Schema order is not preserved by JSONSerialization, so emit keys in a stable order:
    // required first, then the rest, each alphabetically. Any document the decoder produces
    // is valid; it simply cannot reorder keys.
    let names = properties.keys.sorted {
      let leftRequired = required.contains($0)
      let rightRequired = required.contains($1)
      return leftRequired == rightRequired ? $0 < $1 : leftRequired
    }

    var documents = [""]
    for name in names {
      guard let child = properties[name] as? [String: Any] else {
        throw Unsupported.malformed("property \(name) is not a schema")
      }
      let values = try expand(child, limit: limit)
      let optional = !required.contains(name)

      var next: [String] = []
      next.reserveCapacity(documents.count * (values.count + (optional ? 1 : 0)))
      for document in documents {
        if optional { next.append(document) }
        for value in values {
          let pair = "\"\(name)\":" + value
          next.append(document.isEmpty ? pair : document + "," + pair)
        }
      }
      if next.count > limit { throw Unsupported.tooManyDocuments(limit) }
      documents = next
    }
    return documents.map { "{" + $0 + "}" }
  }

  private static func render(_ value: Any) -> String {
    if let text = value as? String {
      let data = try? JSONSerialization.data(
        withJSONObject: [text], options: [.withoutEscapingSlashes])
      if let data, var rendered = String(data: data, encoding: .utf8) {
        rendered.removeFirst()
        rendered.removeLast()
        return rendered
      }
      return "\"\(text)\""
    }
    if let flag = value as? Bool { return flag ? "true" : "false" }
    if let number = value as? NSNumber { return number.stringValue }
    if value is NSNull { return "null" }
    return "null"
  }
}
