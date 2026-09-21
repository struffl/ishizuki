// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Finding a Mac without being told where it is. Bonjour covers a shared network; a Tailscale
// name has to be typed once, which is why a browse result and a typed address end up the same
// kind of thing here.

import Foundation
import IshizukiKit
import Network
import Observation

@MainActor
@Observable
public final class LinkBrowser {
  public struct Found: Identifiable, Sendable, Equatable {
    public var id: String
    public var name: String
    public var endpoint: LinkEndpoint
    public var model: String?
    public var version: String?
  }

  public private(set) var found: [Found] = []
  public private(set) var isBrowsing = false
  public private(set) var failure: String?

  private var browser: NWBrowser?

  public init() {}

  public func start(type: String = BonjourService.linkType) {
    guard browser == nil else { return }
    let parameters = NWParameters()
    parameters.includePeerToPeer = true
    let browser = NWBrowser(
      for: .bonjourWithTXTRecord(type: type, domain: nil), using: parameters)
    browser.stateUpdateHandler = { [weak self] state in
      Task { @MainActor in
        switch state {
        case .ready: self?.isBrowsing = true
        case .failed(let error):
          self?.isBrowsing = false
          self?.failure = String(describing: error)
        case .cancelled: self?.isBrowsing = false
        default: break
        }
      }
    }
    browser.browseResultsChangedHandler = { [weak self] results, _ in
      Task { @MainActor in self?.absorb(results) }
    }
    self.browser = browser
    browser.start(queue: .main)
  }

  public func stop() {
    browser?.cancel()
    browser = nil
    isBrowsing = false
  }

  private func absorb(_ results: Set<NWBrowser.Result>) {
    found = results.compactMap { result in
      guard case .service(let name, let type, let domain, _) = result.endpoint else { return nil }
      var model: String?
      var version: String?
      if case .bonjour(let record) = result.metadata {
        model = record["model"]
        version = record["version"]
      }
      return Found(
        id: "\(name).\(type)\(domain)",
        name: name,
        endpoint: .bonjour(name: name, type: type, domain: domain),
        model: model,
        version: version)
    }
    .sorted { $0.name < $1.name }
  }
}
