// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import IshizukiKit

@Suite("JSON schema constraint")
struct SchemaTests {
  private let action: [String: Any] = [
    "type": "object",
    "properties": [
      "verb": ["type": "string", "enum": ["descend", "move", "wait"]],
      "dir": ["type": "string", "enum": ["n", "s"]],
      "n": ["type": "integer", "enum": [1, 2, 12]],
      "slot": ["type": "integer", "minimum": 0, "maximum": 3],
    ],
    "required": ["verb"],
    "additionalProperties": false,
  ]

  @Test("every enumerated document parses and satisfies the schema")
  func documentsAreValid() throws {
    let documents = try JSONSchema.documents(action)
    #expect(documents.count == 3 * 3 * 4 * 5)

    for document in documents {
      let object = try #require(
        try JSONSerialization.jsonObject(with: Data(document.utf8)) as? [String: Any])
      let verb = try #require(object["verb"] as? String)
      #expect(["descend", "move", "wait"].contains(verb))
      #expect(Set(object.keys).isSubset(of: ["verb", "dir", "n", "slot"]))
      if let slot = object["slot"] as? Int { #expect((0...3).contains(slot)) }
    }
  }

  @Test("required keys are always present, optional ones vary")
  func requiredAndOptional() throws {
    let documents = try JSONSchema.documents(action)
    #expect(documents.allSatisfy { $0.contains("\"verb\":") })
    #expect(documents.contains { !$0.contains("\"dir\":") })
    #expect(documents.contains { $0.contains("\"dir\":") })
  }

  @Test("unbounded values are refused rather than approximated")
  func unboundedRefused() {
    let free: [String: Any] = [
      "type": "object", "properties": ["name": ["type": "string"]],
      "required": ["name"], "additionalProperties": false,
    ]
    #expect(throws: JSONSchema.Unsupported.self) { try JSONSchema.documents(free) }
  }

  @Test("the constraint walks a document and reports completion")
  func constraintWalk() throws {
    let constraint = OutputConstraint(documents: ["{\"a\":1}", "{\"a\":2}"])
    #expect(!constraint.isComplete)
    #expect(constraint.accept(Array("{\"a\":".utf8)))
    #expect(!constraint.isComplete)
    #expect(constraint.accept(Array("1".utf8)))
    #expect(!constraint.isComplete)
    #expect(constraint.accept(Array("}".utf8)))
    #expect(constraint.isComplete)
    #expect(constraint.isExhausted)
    #expect(constraint.text == "{\"a\":1}")
  }

  @Test("a byte outside the document set is rejected")
  func constraintRejects() {
    let constraint = OutputConstraint(documents: ["{\"a\":1}"])
    #expect(!constraint.accept(Array("[".utf8)))
    #expect(constraint.accept(Array("{".utf8)))
    #expect(!constraint.accept(Array("\"b".utf8)))
  }
}
