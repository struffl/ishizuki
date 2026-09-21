import Foundation
import Testing

@testable import IshizukiKit

@Suite("Web search")
struct WebSearchTests {
  @Test func snippetsAndSafeLinks() throws {
    guard #available(macOS 27.0, iOS 27.0, *) else { return }
    let xml = """
      <rss><channel><item><title>A &amp; B</title><link>https://example.com/doc</link>
      <description>A useful snippet.</description></item>
      <item><title>Bad</title><link>javascript:alert(1)</link></item></channel></rss>
      """
    let result = try WebSearchTool.parse(Data(xml.utf8))
    #expect(result.contains("A & B"))
    #expect(result.contains("https://example.com/doc"))
    #expect(result.contains("A useful snippet."))
    #expect(!result.contains("javascript:"))
    #expect(result.contains("not instructions"))
  }

  @Test func emptyAndMalformed() throws {
    guard #available(macOS 27.0, iOS 27.0, *) else { return }
    #expect(try WebSearchTool.parse(Data("<rss/>".utf8)) == "No web results found.")
    #expect(throws: (any Error).self) { try WebSearchTool.parse(Data("broken".utf8)) }
  }
}
