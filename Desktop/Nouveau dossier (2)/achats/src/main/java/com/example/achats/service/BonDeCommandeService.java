package com.example.achats.service;

import com.example.achats.model.BonDeCommande;
import com.example.achats.repository.BonDeCommandeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class BonDeCommandeService {

    @Autowired
    private BonDeCommandeRepository bonDeCommandeRepository;

    public List<BonDeCommande> findAll() {
        return bonDeCommandeRepository.findAll();
    }

    public BonDeCommande findById(Long id) {
        return bonDeCommandeRepository.findById(id).orElse(null);
    }

    public BonDeCommande save(BonDeCommande bonDeCommande) {
        return bonDeCommandeRepository.save(bonDeCommande);
    }

    public void deleteById(Long id) {
        bonDeCommandeRepository.deleteById(id);
    }
}
